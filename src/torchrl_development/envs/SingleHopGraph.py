
# import
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
from torchrl_development.envs.utils import TimeAverageStatsCalculator, create_discrete_rv, create_poisson_rv, FakeRV

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class SingleHopGraph(EnvBase):
    """
    Internal state is Q and Y (queues and link states) which are both (N+1)x1 vectors
    where N is the number of nodes in the network.
    The first element of Q is the destination node (which should always be zero)
    The first element of Y can be considered a self-loop from destination to destination

    Actions are one-hot vectors of length N+1
    If the first element is 1, the action is to idle

    The state transitions are guided by the following

    Q(t+1) = max(Q(t) - Y(t) * A, 0) + X(t)

    where A is the action vector and X(t) is the arrival vector

    The reward is the negative of the sum of the elements of Q at time t, meaning before the action and new arrivals

    X(t) is sampled from a set of arrival distributions, which is a list of N+1 distributions (generators)
    Y(t) is sampled from a set of link state distributions, which is a list of N+1 distributions (generators)
    First element of X_gen is the destination node, which should always be zero
    First element of Y_gen is will always be 1, meaning we can always idle


    """


    batch_locked = False
    def __init__(self, net_para, seed = None, device = "cpu"):
        super().__init__()
        # Seeding
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self._set_seed(seed)

        self.env_id = net_para.get("env_id", "SingleHop")

        # Nodes/Buffers/Queues
        self.nodes = net_para['nodes']
        self.N = len(self.nodes)-1
        # Destination - always the last node
        self.destination = 0
        # Initialize the buffers for each node
        self.Q = np.zeros((len(self.nodes),))
        # Set episode truncation condition
        self.terminal_backlog = net_para.get("terminal_backlog", None) # if the sum of the queues is greater than this, the episode is terminated with a large negative reward

        # Check if net_para has 'arrival_distribution' and 'service_distribution' keys
        if 'arrival_distribution' in net_para.keys():
            self.arrival_distribution = net_para['arrival_distribution']
        else:
            self.arrival_distribution = 'discrete'
        if 'service_distribution' in net_para.keys():
            self.service_distribution = net_para['service_distribution']
        else:
            self.service_distribution = 'discrete'

        # Get edge_indices
        self._extract_edge_indices()


        # Extract X_gen and X_map
        self._extract_X_gen(net_para['X_params'])
        # Extract Y_gen and Y_map
        self._extract_Y_gen(net_para['Y_params'])
        # Compute load
        self._compute_load()
        self.debug = net_para.get("debug", False)

        # Whether or not the state includes arrival rates
        self.obs_lambda = net_para.get("obs_lambda", False)

        # Add baseline lta performance for maxweight
        self.baseline_lta = net_para.get("baseline_lta", None)

        # Track standard deviation of the time average
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


    def _extract_edge_indices(self):
        """
        We are dealing with fully connected graphs, so this should be every combination of nodes
        :return:
        """
        adj = torch.ones([self.N,1]) @ torch.ones([1,self.N]) - torch.eye(self.N)
        self.edge_indices = adj.to_sparse().indices()

    def _extract_X_gen(self, X_params):
        # creates generators for the arrival distributions
        # in this case, the number of entries is equal to the number of nodes + 1
        X_map = OrderedDict({"0": {"source": 0, "destination": 0}}) # 0 corresponds to the destination node/basestation
        X_gen = [FakeRV(0)] # Create placeholder for the destination node
        self.arrival_rates = []
        for key, value in X_params.items():
            if self.arrival_distribution == 'discrete':
                X_gen.append(create_discrete_rv(self.np_rng, nums = value["arrival"], probs = value['probability']))
            elif self.arrival_distribution == 'poisson':
                X_gen.append(create_poisson_rv(self.np_rng, rate = value['arrival_rate']))
            X_map[key] = {"source": value['source'], "destination": value['destination']}
            self.arrival_rates.append(X_gen[-1].mean())
        self.X_map = X_map
        self.X_gen = X_gen

    def _gen_arrivals(self):
        """
        Simulates the arrivals for each sources
        :return: total number of arrivals
        """
        # Sample X_gen for each source
        X = np.array([gen.sample() for gen in self.X_gen])
        # Increment the corresponding buffer
        self.Q = np.add(self.Q, X)
        return X.sum()


    def _extract_Y_gen(self, Y_params):
        # creates generators for the link state distributions
        Y_map = OrderedDict({"0": {"source": 0, "destination": 0}})
        Y_gen = [FakeRV(0)]
        link_rates = []
        for key, value in Y_params.items():
            if self.service_distribution == 'discrete':
                Y_gen.append(create_discrete_rv(self.np_rng, nums = value['capacity'], probs = value['probability']))
            elif self.service_distribution == 'poisson':
                Y_gen.append(create_poisson_rv(self.np_rng, rate = value['service_rate']))
            Y_map[key] = {"source": value['source'], "destination": value['destination']}
            link_rates.append(Y_gen[-1].mean())
        self.service_rates = link_rates
        self.link_rates = link_rates
        self.Y_map = Y_map
        self.Y_gen = Y_gen
        self._gen_link_states()


    def _compute_load(self):
        "Computes the load of the network defined as the average ratio of arrivals to the capacities"
        self.network_load = np.sum([x/y for x,y in zip(self.arrival_rates, self.link_rates)])



    def _gen_link_states(self):
        """
        Simulates the link states for each link
        :return: True if there is a connected link, False otherwise
        """
        # Sample X_gen for each source
        link_states = np.array([gen.sample() for gen in self.Y_gen[1:]])
        # idle state is only one if all link states are zero
        idle_state = np.ones([1])*np.all(link_states == 0)
        self.Y = np.concatenate([idle_state, link_states], axis = 0)

    def get_mask(self):
        # Mask 1
        mask1 = self.Q[1:]*self.Y[1:] != 0 # evaluates to true if the queue is empty or the link is disconnected
        # Mask 2
        mask2 = (mask1 == False).all().reshape(1) # evaluates to true if all queues are empty or all links are disconnected

        return np.concatenate([mask2, mask1], axis = 0)

    def get_Y_out(self):
        return self.Y[1:]

    def get_Q_out(self):
        return self.Q[1:]

    def get_backlog(self):
        return np.sum(self.Q)

    def _get_reward(self):
        return -self.get_backlog()

    def get_np_obs(self):
        return np.concatenate([self.get_Q(), self.get_Y()])


    def _make_spec(self):

        self.graph_spec = CompositeSpec(
            x = UnboundedContinuousTensorSpec(
                shape = (len(self.nodes)-1, 2), # might want to change the dimensions to add the context as the observation
                dtype = torch.float
            ),
            edge_index = UnboundedContinuousTensorSpec(
                shape = self.edge_indices.shape,
                dtype = torch.long
            ),
            shape = ())

        self.observation_spec= CompositeSpec(
            Q = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.int
            ),
            Y = BoundedTensorSpec(
                low=0,
                high=100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.int
            ),
            backlog = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = 1,
                dtype = torch.int
            ),
            mask = DiscreteTensorSpec(
                n = len(self.nodes),
                shape = (len(self.nodes),),
                dtype = torch.bool
            ),
            graph = self.graph_spec,
            shape = (),

        )
        # if self.obs_lambda, add the arrival rates to the observation spec
        if self.obs_lambda:
            self.observation_spec.set("lambda", BoundedTensorSpec(
                low =0,
                high = 10,
                shape=(len(self.nodes) - 1,),
                dtype=torch.float
            ))
        if self.track_stdev:
            self.observation_spec.set("ta_stdev", UnboundedContinuousTensorSpec(
                shape = (1,),
                dtype = torch.float
            ))

        self.action_spec = OneHotDiscreteTensorSpec(
            n = len(self.nodes), # 0 is idling, n corresponds to node n
            shape = (len(self.nodes),),
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




    def _step(self, tensordict: TensorDict):

        action = tensordict["action"].numpy()
        # see if action is invalid by checking if the sum == 1
        if np.sum(action) != 1:
            raise ValueError("Action must be a one-hot vector")


        # Step 1: Get the corresponding reward
        reward = self._get_reward()

        # Step 1: Apply the action
        if action[0] is True:
            # idle
            pass
        else:
            self.Q = np.max([self.Q - np.multiply(self.Y,action), np.zeros(len(self.nodes))], axis = 0)

        if (self.Q < 0).any():
            raise ValueError("Queue length cannot be negative")


        # Step 3: Generate New Arrivals
        n_arrivals = self._gen_arrivals()

        # Step 4: Generate new capacities
        self._gen_link_states()

        # Step 5: Get backlog for beginning of s_{t+1} and update running stats

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
                {"Q": torch.tensor(self.get_Q_out(), dtype = torch.int),
                "Y": torch.tensor(self.get_Y_out(), dtype = torch.int),
                "truncated": torch.tensor(truncate, dtype = torch.bool),
                "terminated": torch.tensor(terminated, dtype = torch.bool),
                "reward": torch.tensor(reward, dtype = torch.float),
                "backlog": torch.tensor(next_backlog, dtype = torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype = torch.bool),
                "graph": self.get_graph_state()
                }, batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.arrival_rates, dtype=torch.float))
        if self.track_stdev:
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
        return out

    def get_graph_state(self):
        return TensorDict({
            "x": torch.tensor(np.stack([self.get_Q_out(), self.get_Y_out()], axis = 1), dtype = torch.float),
            "edge_index": torch.tensor(self.edge_indices, dtype = torch.long),
        }, batch_size=[])


    def _reset(self, tensordict: TensorDict):

        #np.random.seed(seed)
        # make value =0 for all keys in self.q_state
        self.Q = np.zeros((len(self.nodes),))
        self._gen_link_states()
        out = TensorDict({"Q": torch.tensor(self.get_Q_out(), dtype = torch.int),
                "Y": torch.tensor(self.get_Y_out(), dtype = torch.int),
                "done": torch.tensor(False, dtype = torch.bool),
                "terminated": torch.tensor(False, dtype = torch.bool),
                "truncated": torch.tensor(False, dtype = torch.bool),
                "backlog": torch.tensor(self.get_backlog(), dtype=torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype=torch.bool),
                "graph": self.get_graph_state()
                          },
                         batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.arrival_rates, dtype=torch.float))
        if self.track_stdev:
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
        return out

    def _set_seed(self, seed: Optional[int]):
        self.rng = torch.manual_seed(seed)
        self.np_rng = np.random.default_rng(seed)
        self.seed = seed
    def _debug_printing(self, init_buffer, current_capacities, delivered,
                           ignore_action, action, post_action_buffer,
                           post_arrival_buffer, n_arrivals, reward, new_capacities):
        print("="*20)
        print(f"Initial Buffer: {init_buffer}")
        print(f"Current Capacities: {current_capacities}")
        print(f"Ignore Action: {ignore_action}")
        print(f"Action: {action}")
        print(f"Delivered: {delivered}")
        print(f"Post Action Buffer: {post_action_buffer}")
        print(f"Reward: {reward}")
        print("Arrivals: ", n_arrivals)
        print(f"Post Arrival Buffer: {post_arrival_buffer}")
        print(f"New Capacities: {new_capacities}")
        print("="*20)
        print("\n")



