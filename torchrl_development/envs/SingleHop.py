
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
from torchrl_development.envs.utils import TimeAverageStatsCalculator, create_discrete_rv, create_poisson_rv, FakeRV, create_uniform_rv

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



class SingleHop(EnvBase):
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
        self.arrival_distribution = net_para.get('arrival_distribution', 'discrete')
        self.service_distribution = net_para.get('service_distribution', 'discrete')

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
            elif self.arrival_distribution == 'uniform':
                X_gen.append(create_uniform_rv(self.np_rng, high = value['arrival_rate']*2))
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
        return X


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
            elif self.service_distribution == 'uniform':
                Y_gen.append(create_uniform_rv(self.np_rng, high = value['service_rate']*2))
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

    def set_state(self, state_array: np.array):
        # Sets the Q and Y state based on the input array
        self.Q = np.concatenate([np.zeros(1), state_array[:self.N]])
        self.Y = np.concatenate([np.zeros(1), state_array[self.N:]])

    def _get_reward(self):
        return -self.get_backlog()

    def get_np_obs(self):
        return np.concatenate([self.get_Q(), self.get_Y()])


    def _make_spec(self):


        self.observation_spec= CompositeSpec(
            Q = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.float
            ),
            Y = BoundedTensorSpec(
                low=0,
                high=100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.float
            ),
            departures = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.float
            ),
            arrivals = BoundedTensorSpec(
                low = 0,
                high = 100_000,
                shape = (len(self.nodes)-1,),
                dtype = torch.float
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
        if action[0] is True:
            # idle
            pass
        else:
            self.Q = np.max([self.Q - np.multiply(self.Y,action), np.zeros(len(self.nodes))], axis = 0)

        if (self.Q < 0).any():
            raise ValueError("Queue length cannot be negative")

        # Measure departures from the queues
        departures = (self.Q - old_Q)[1:]


        # Step 3: Generate New Arrivals
        arrivals = self._gen_arrivals()[1:]

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
                {"Q": torch.tensor(self.get_Q_out(), dtype = torch.float),
                "Y": torch.tensor(self.get_Y_out(), dtype = torch.float),
                "departures": torch.tensor(departures, dtype = torch.float),
                "arrivals": torch.tensor(arrivals, dtype = torch.float),
                "truncated": torch.tensor(truncate, dtype = torch.bool),
                "terminated": torch.tensor(terminated, dtype = torch.bool),
                "reward": torch.tensor([reward], dtype = torch.float),
                "backlog": torch.tensor(next_backlog, dtype = torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype = torch.bool),
                }, batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.arrival_rates, dtype=torch.float))
        if self.track_stdev:
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
        if getattr(self, "baseline_lta", None) is not None:
            out.set("baseline_lta", torch.tensor(self.baseline_lta, dtype = torch.float))
        if getattr(self, "context_id", None) is not None:
            out.set("context_id", torch.tensor(self.context_id, dtype = torch.int))
        return out

    def _reset(self, tensordict: TensorDict):

        #np.random.seed(seed)
        # make value =0 for all keys in self.q_state
        self.Q = np.zeros((len(self.nodes),))
        self._gen_link_states()
        out = TensorDict(
        {"Q": torch.tensor(self.get_Q_out(), dtype = torch.float),
                "Y": torch.tensor(self.get_Y_out(), dtype = torch.float),
                "departures": torch.tensor(np.zeros(len(self.nodes)-1), dtype = torch.float),
                "arrivals": torch.tensor(np.zeros(len(self.nodes)-1), dtype = torch.float),
                "done": torch.tensor(False, dtype = torch.bool),
                "terminated": torch.tensor(False, dtype = torch.bool),
                "truncated": torch.tensor(False, dtype = torch.bool),
                "backlog": torch.tensor(self.get_backlog(), dtype=torch.int).reshape(1),
                "mask": torch.tensor(self.get_mask(), dtype=torch.bool),
                          },
                         batch_size=[])
        if self.obs_lambda:
            out.set("lambda", torch.tensor(self.arrival_rates, dtype=torch.float))
        if self.track_stdev:
            self.time_avg_stats.reset()
            out.set("ta_stdev", torch.Tensor([self.time_avg_stats.sampleStdev]))
        if getattr(self, "baseline_lta", None) is not None:
            out.set("baseline_lta", torch.tensor(self.baseline_lta, dtype=torch.float))
        if getattr(self, "context_id", None) is not None:
            out.set("context_id", torch.tensor(self.context_id, dtype=torch.int))
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


    def get_stable_action(self, type = "LQ"):
        q_obs = self.get_buffers()
        if type == "LQ":
            action = np.random.choice(np.where(q_obs == q_obs.max())[0]) + 1  # LQ
        elif type == "SQ":
            q_obs[q_obs == 0] = 1000000
            action = np.random.choice(np.where(q_obs == q_obs.min())[0]) + 1
        elif type == "RQ":
            action = np.random.choice(np.where(q_obs > 0)[0]) + 1
        elif type == "LCQ":
            cap = self.get_cap()
            connected_obs = cap * q_obs
            action = np.random.choice(np.where(connected_obs == connected_obs.max())[0]) + 1
        elif type == "MWQ": #Max Weighted Queue
            p_cap = self.service_rate
            valid_obs = q_obs > 0
            # choose the most reliable queue that is not empty
            valid_p_cap = p_cap * valid_obs**2
            if np.all(valid_obs == False):
                action = 0
            else:
                action = np.random.choice(np.where(valid_p_cap == valid_p_cap.max())[0]) + 1
        elif type == "MWCQ": #Max Weighted Connected Queue
            cap = self.get_cap()
            weighted_obs = cap * q_obs**2
            if np.all(weighted_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(weighted_obs == weighted_obs.max())[0]) + 1

        elif type == "RCQ": # Random Connected Queue
            cap = self.get_cap()
            connected_obs = cap * q_obs
            if np.all(connected_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(connected_obs > 0)[0]) + 1
        elif type == "LRCQ": # Least Reliable Connected Queue
            p_cap = self.unreliabilities
            cap = self.get_cap()
            connected_obs = cap * q_obs
            weighted_obs = p_cap * connected_obs
            if np.all(weighted_obs == 0):
                action = 0
            else:
                action = np.random.choice(np.where(weighted_obs == weighted_obs.max())[0]) + 1
        elif type == "Optimal":
            if not self.obs_links:
                p_cap = 1 - self.unreliabilities
                non_empty = q_obs > 0
                action  = np.argmax(p_cap*non_empty)+1
            else:
                # should select the connected link with the lowest success probability
                p_cap = 1 - self.unreliabilities
                non_empty = q_obs > 0
                action = np.argmax(p_cap * non_empty) + 1


        if not isinstance(action, int):
            action = action.astype(int)
        return action

    def get_serviceable_buffers(self):
        if self.obs_links:
            temp = self.get_buffers() * self.get_cap()
            return temp



    # def get_mask(self, state = None):
    #     """
    #     Cases:
    #     Based on buffers being empty
    #         1) All buffers are empty -> mask all but action 0
    #         2) There is at least one non-empty buffer -> mask action 0 and all empty buffers
    #     Based on links being connected AND observed
    #     1.) Mask the actions corresponding to non-connected links
    #         - If already masked, leave masked, if not masked but not connected mask
    #             mask[1:] = np.logical_or(mask[1:], 1-self.get_cap())
    #     Returns: Boolean mask vector corresponding to actions 0 to n_queues
    #     -------
    #
    #     """
    #     # returns a vector of length n_queues + 1
    #     mask = np.bool_(np.zeros(self.n_queues+1)) # size of the action space
    #     # masking based on buffers being empty
    #     if self.get_backlog() == 0:
    #         mask[1:] = True
    #     else: # Case 2
    #         mask[0] = True # mask action 0
    #         mask[1:] = self.get_buffers() < 1 # mask all empty buffers
    #     # masking based on connected links
    #     if self.obs_links:
    #         mask[1:] = np.logical_not(self.get_serviceable_buffers())
    #         if np.all(mask[1:]==True):
    #             mask[0] = False
    #     if np.all(mask):
    #         raise ValueError("Mask should not be all True")
    #     elif np.all(mask == False):
    #         raise ValueError("Mask should not be all False")
    #     return mask


    def _sim_arrivals(self):
        n_arrivals = 0
        for cls_num, cls_rv in enumerate(self.classes.items()):
            source = cls_num+1
            if cls_rv.sample() == 1:
                self.buffers[source] += 1
                n_arrivals += 1
        return n_arrivals

    # def _extract_capacities(self, cap_dict):
    #     caps = {}
    #     # if '(0,0)' in cap_dict.keys():
    #     #     # All links have the same capacity and probability
    #     #     capacity = cap_dict['(0,0)']['capacity']
    #     #     probability = cap_dict['(0,0)']['probability']
    #     #     for link in self.links:
    #     #         caps[link] = create_rv(self.rng, num=capacity, prob = probability)
    #
    #     for link, l_info in cap_dict.items():
    #         if isinstance(link, str):
    #             link = eval(link)
    #         if link == (0,0):
    #             continue
    #         capacity = eval(l_info['capacity']) if isinstance(l_info['capacity'], str) else l_info['capacity']
    #         probability = eval(l_info['probability']) if isinstance(l_info['probability'], str) else l_info['probability']
    #
    #         rv = create_rv(nums = capacity, probs = probability)
    #         caps[link] = rv
    #
    #         # generate unreliabilities
    #     service_rate = []
    #     for link in self.links:
    #         if link[1] == self.destination:
    #             service_rate.append(caps[link].mean())
    #     self.service_rate = np.array(service_rate)
    #
    #
    #     if (0,0) in caps.keys():
    #         del caps[(0,0)]
    #
    #     # Convert caps to a list
    #     caps = np.array(list(caps.values()))
    #
    #     return caps
    #
    # def _extract_classes(self, class_dict):
    #     classes = []
    #     destinations = []
    #     for cls_num, cls_info in class_dict.items():
    #         rv = create_rv(nums =[0, cls_info['arrival']], probs = [1- cls_info["probability"], cls_info['probability']])
    #         classes.append(rv)
    #     if len(classes) != len(list(self.links)):
    #         raise ValueError("Number of classes must equal number of links")
    #
    #     return classes, destinations

    # def _sim_arrivals(self):
    #     arrivals = np.zeros(len(self.classes))
    #     for cls_num, cls_rv in enumerate(self.classes):
    #         source = cls_num+1
    #         if cls_rv.sample() == 1:
    #             self.buffers[source] += 1
    #             arrivals[source-1] += 1
    #     return arrivals
    #
    # def set_state(self, state):
    #     if self.obs_links:
    #         for i in range(self.n_queues):
    #             self.buffers[i+1] = state[i]
    #         for j in range(self.n_queues):
    #             self.Cap[j] = state[j+self.n_queues]
    #
    #     return self.get_obs()
    #
    # def estimate_transitions(self, state, action, max_samples = 1000, min_samples = 100, theta = 0.001):
    #
    #     C_sas = {} # counter for transitions (s,a,s')
    #     P_sas = {} # probabilities for transitions (s,a,s')
    #     diff = {}
    #     for n in range(1, max_samples+1):
    #         self.set_state(state)
    #         next_state, _, _, _, _ = self.step(action)
    #         if tuple(next_state) in C_sas.keys():
    #             C_sas[tuple(next_state)] += 1# increment the counter for the visits to next state
    #             p_sas = C_sas[tuple(next_state)]/n # calculate the probability of the transition
    #             diff[tuple(next_state)] = np.abs(P_sas[tuple(next_state)] - p_sas)
    #             P_sas[tuple(next_state)] = p_sas
    #         else:
    #             C_sas[tuple(next_state)] = 1
    #             P_sas[tuple(next_state)] = 1
    #
    #         # check for convergence:
    #         if n > min_samples and np.all(list(diff.values())) < theta:
    #             break
    #     P_sas = {key: value/n for key, value in C_sas.items()}
    #     if np.abs(np.sum(list(P_sas.values())) - 1) > 0.001:
    #         raise ValueError("Transition Probabilities do not sum to one")
    #
    #     # convert to probabilities
    #
    #     return P_sas, n
