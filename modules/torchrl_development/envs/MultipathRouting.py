from copy import deepcopy
from typing import Optional
from collections import OrderedDict

import numpy as np
import torch
from tensordict import TensorDict

from torchrl.data import BoundedTensorSpec, CompositeSpec, OneHotDiscreteTensorSpec, UnboundedContinuousTensorSpec, DiscreteTensorSpec
from torchrl.envs import (
    EnvBase,
)
from modules.torchrl_development.envs.utils import TimeAverageStatsCalculator, create_discrete_rv, create_poisson_rv, FakeRV, create_uniform_rv

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class MultipathRouting(EnvBase):
    """
    Internal state is Q and Y (queue length and service rate) for  each server which are both K x 1 vectors
    where K is the number of servers.

    At each time-step there is a random arrival x_{t}\sim P(X)

    Actions are one-hot vectors of length K, where if the j-th element is one x_t packets are routed to server
    j

    The state-transitions are guided by the following:

    Q(t+1) = max(Q_t-Y_t,0)+A_t\times x_t

    The reward is the negative sum of the queue lengths at each server at time t

    """

    def __init__(self, net_para, seed = None):
        super().__init__()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)
        self.env_id = net_para.get("env_id", "MultipathRouting")

        # Servers/Queues
        self.servers = net_para["servers"]
        self.K = self.N = self.servers
        self.Q = np.zeros([self.K,])
        self.Y = torch.zeros([self.K,])
        self.terminal_backlog = net_para.get("terminal_backlog", None)

        # Check if net_para has 'arrival_distribution' and 'service_distribution' keys
        self.arrival_distribution = net_para.get("arrival_distribution", "poisson")
        self.service_distribution = net_para.get("service_distribution", "poisson")

        # create random variables for the arrival and service distributions
        self._extract_X_gen(net_para["X_params"])
        self._extract_Y_gen(net_para["Y_params"])

        # Whether or not the state includes arrival rates
        self.obs_lambda = net_para.get("obs_lambda", True) # lambda is the

        # Whether or not the state includes capacity rates
        self.obs_mu = net_para.get("obs_mu", True)

        # Add baseline lta performance for stable algorithm
        self.baseline_lta = net_para.get("baseline_lta", None)

        # Add context_id if available
        self.context_id = net_para.get("context_id", None)

        self.terminate_on_lta_threshold = net_para.get("terminate_on_lta_threshold", False)
        self.terminal_lta_factor = net_para.get("terminal_lta_factor", 5)

        # Tracking running state
        self.time_avg_stats = TimeAverageStatsCalculator(net_para.get("stat_window_size", 10000))
        self.terminate_on_convergence = net_para.get("terminate_on_convergence", False)
        self.convergence_threshold = net_para.get("convergence_threshold", 0.05)
        self.track_stdev = True

        # Create specs for torchrl
        self._make_spec()
        self._make_done_spec()


    def _step(self, tensordict: TensorDict):

        action = tensordict["action"].squeeze().cpu().numpy()

        # see if action is invalid by checking if the sum == 1
        if np.sum(action) > 1:
            # How to deal with tie?
            zero_action = np.zeros_like(action)
            chosen_action = np.random.choice(np.where(action == action.max())[0])
            action = zero_action
            action[chosen_action] = 1
            # action_values = tensordict["action_value"]
            # raise ValueError(f"Action must be a one-hot vector - instead got {action} \\"
            #                  f"with values {action_values}")


        # make sure the action is the same shape as Y


        # Step 1: Get the corresponding reward
        reward = self._get_reward()

        # Record current self.Q to compute difference after applying action
        old_Q = self.Q.copy()

        # Step 1: Apply the action
        self.Q = np.max([self.Q - self.Y, np.zeros((self.K,))], axis = 0) + self.X * action

        if (self.Q < 0).any():
            raise ValueError("Queue length cannot be negative")

        # Measure departures from the queues
        departures = (self.Q - old_Q)


        # Step 3: Generate New Arrivals
        self.X = self._gen_arrivals()

        # Step 4: Generate new service rates
        self.Y = self._gen_service_rates()

        # Step 5: Get backlog for beginning of s_{t+1} and update running states
        next_backlog = self.get_backlog()
        self.time_avg_stats.update(next_backlog)

        # Step 6: Check if the episode should be truncated
        if self.terminal_backlog is not None:
            truncate = next_backlog > self.terminal_backlog
            reward = -100 if truncate else reward
        else:
            truncate = False
        if self.baseline_lta is not None and self.terminate_on_lta_threshold:
            if self.time_avg_stats.mean >  self.terminal_lta_factor * self.baseline_lta:
                truncate = True

        # Step 7: Check if the episode should be truncated
        terminated = False
        if self.terminate_on_convergence and self.time_avg_stats.is_full: # first check if we have enough data
            if self.time_avg_stats.sampleStdev < self.convergence_threshold:
                terminated = True


        out = TensorDict(
                {"Q": torch.tensor(self.Q, dtype = torch.float32),
                "Y": torch.tensor(self.Y, dtype = torch.float32),
                "departures": torch.tensor(departures, dtype = torch.float32),
                "arrivals": torch.tensor(self.X, dtype = torch.float32)*torch.ones([self.K,], dtype = torch.float32),
                "truncated": torch.tensor(truncate, dtype = torch.bool),
                "terminated": torch.tensor(terminated, dtype = torch.bool),
                "reward": torch.tensor([reward], dtype = torch.float32),
                "backlog": torch.tensor(next_backlog, dtype = torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype = torch.bool),
                }, batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.lam, dtype=torch.float32)*torch.ones([self.K,], dtype = torch.float32))
        if self.obs_mu:
            out.set("mu", self.mu)
        if self.track_stdev:
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
            out.set("ta_mean", torch.Tensor([self.time_avg_stats.mean]))
        if getattr(self, "baseline_lta", None) is not None:
            out.set("baseline_lta", torch.tensor(self.baseline_lta, dtype = torch.float))
        if getattr(self, "context_id", None) is not None:
            out.set("context_id", torch.tensor(self.context_id, dtype = torch.int))
        return out

    def _reset(self, tensordict: TensorDict):

        #np.random.seed(seed)
        # make value =0 for all keys in self.q_state
        self.Q = np.zeros(self.K,)
        self.Y = self._gen_service_rates()
        self.X = self._gen_arrivals()
        out = TensorDict(
        {"Q": torch.tensor(self.Q, dtype = torch.float32),
                "Y": torch.tensor(self.Y, dtype = torch.float32),
                "departures": torch.tensor(np.zeros(self.K), dtype = torch.float32),
                "arrivals": torch.tensor(self.X, dtype = torch.float32)*torch.ones([self.K,], dtype = torch.float32),
                "done": torch.tensor(False, dtype = torch.bool),
                "terminated": torch.tensor(False, dtype = torch.bool),
                "truncated": torch.tensor(False, dtype = torch.bool),
                "backlog": torch.tensor(self.get_backlog(), dtype=torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype=torch.bool),
                          },
                         batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.lam, dtype=torch.float32)*torch.ones([self.K,], dtype = torch.float32))
        if self.obs_mu:
            out.set("mu", torch.tensor(self.mu, dtype=torch.float32))
        if self.track_stdev:
            # self.time_avg_stats.reset()
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
            out.set("ta_mean", torch.Tensor([self.time_avg_stats.mean]))
        if getattr(self, "baseline_lta", None) is not None:
            out.set("baseline_lta", torch.tensor(self.baseline_lta, dtype=torch.float))
        if getattr(self, "context_id", None) is not None:
            out.set("context_id", torch.tensor(self.context_id, dtype=torch.int))
        return out


    def _get_reward(self):
        return -self.get_backlog()

    def get_backlog(self):
        return np.sum(self.Q)

    def get_mask(self):
        return np.ones([self.K])
    # Private methods
    def _make_spec(self):


        self.observation_spec= CompositeSpec(
            Q = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (self.K,),
                dtype = torch.float
            ),
            Y = BoundedTensorSpec(
                low=0,
                high=100_000,
                shape = (self.K,),
                dtype = torch.float
            ),
            departures = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (self.K,),
                dtype = torch.float
            ),
            arrivals = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (1,),
                dtype = torch.float
            ),
            backlog = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = 1,
                dtype = torch.int
            ),
            mask = DiscreteTensorSpec(
                n = self.K,
                shape = (self.K,),
                dtype = torch.bool
            ),
            # truncated = DiscreteTensorSpec(
            #     n= 2,
            #     dtype = torch.bool
            # ),
            shape = (),

        )
        # if self.obs_lambda, add the arrival rates to the observation spec
        if self.obs_lambda:
            self.observation_spec.set("lambda", BoundedTensorSpec(
                low =0,
                high = 10,
                shape=(1,),
                dtype=torch.float
            ))
        if self.obs_mu:
            self.observation_spec.set("mu", BoundedTensorSpec(
                low = 0,
                high = 1000,
                shape=(self.K,),
                dtype=torch.float
            ))
        if self.track_stdev:
            self.observation_spec.set("ta_stdev", UnboundedContinuousTensorSpec(
                shape = (1,),
                dtype = torch.float
            ))
            self.observation_spec.set("ta_mean", UnboundedContinuousTensorSpec(
                shape = (1,),
                dtype = torch.float
            ))

        if getattr(self, "baseline_lta", None) is not None:
            self.observation_spec.set("baseline_lta", UnboundedContinuousTensorSpec(
                shape = (1,),
                dtype = torch.float
            ))

        if getattr(self, "context_id", None) is not None:
            self.observation_spec.set("context_id",  UnboundedContinuousTensorSpec(
                shape = (1,),
                dtype = torch.float
            ))

        self.action_spec = OneHotDiscreteTensorSpec(
            n = self.K, # 0 is idling, n corresponds to node n
            shape = (self.K,),
        )
        self.reward_spec = UnboundedContinuousTensorSpec(
            shape = (1,),
            dtype = torch.float
        )
    def _make_done_spec(self):  # noqa: F811
        self.done_spec = CompositeSpec(
            {
                "done": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape = (1,)
                ),
                "terminated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape = (1,)
                ),
                "truncated": DiscreteTensorSpec(
                    2, dtype=torch.bool, device=self.device, shape = (1,)
                ),
            },
            shape=self.batch_size,
        )


    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed

    def _extract_X_gen(self, X_params):
        # creates generators for the arrival distributions
        # in this case, the number of entries is equal to the number of nodes + 1
        if self.arrival_distribution == 'discrete':
            X_gen = create_discrete_rv(self.np_rng, nums = X_params["arrival"], probs = X_params['probability'])
        elif self.arrival_distribution == 'poisson':
            X_gen = create_poisson_rv(self.np_rng, rate = X_params['arrival_rate'])
        elif self.arrival_distribution == 'uniform':
            X_gen = create_uniform_rv(self.np_rng, high = X_params['arrival_rate']*2)
        self.lam = np.array([X_gen.mean()], dtype=np.float32).reshape([1,])
        self.X_gen = X_gen

    def _gen_arrivals(self):
        return self.X_gen.sample()

    def _extract_Y_gen(self, Y_params):
        Y_gen = []
        mu = []
        for key, value in Y_params.items():
            if self.service_distribution == 'discrete':
                Y_gen.append(create_discrete_rv(self.np_rng, nums = value["service"], probs = value['probability']))
            elif self.service_distribution == 'poisson':
                Y_gen.append(create_poisson_rv(self.np_rng, rate = value['service_rate']))
            elif self.service_distribution == 'uniform':
                Y_gen.append(create_uniform_rv(self.np_rng, high = value['service_rate']*2))
            mu.append(Y_gen[-1].mean())
        self.Y_gen = Y_gen
        self.Y = self._gen_service_rates()
        self.mu = np.array(mu, dtype=np.float32).reshape([self.K,])

    def _gen_service_rates(self):
        return np.array([gen.sample() for gen in self.Y_gen])


    def get_stable_action(self, td = None, type = "SQ"):
        if type == "SQ": #return argmin of Q as a one-hot vector
            if td is None:
                return np.eye(self.K)[np.argmin(self.Q)]
            else:
                td["action"] = np.eye(self.K)[np.argmin(self.Q)]
                return td
        else:
            raise ValueError("Invalid type")

    def set_state(self, state: list):
        # input will be a list of length 2K, the first K elements will be the queue lengths and the next K elements will be the service rates
        self.Q = np.array(state[:self.K])
        self.Y = np.array(state[self.K:])

