
# import
from copy import deepcopy
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import networkx as nx

import numpy as np
import torch
from tensordict import TensorDict, merge_tensordicts

from torchrl.data import Composite, Bounded, Unbounded, Binary

from torchrl.envs import (
    EnvBase,
)

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class FakeRV:

    def __init__(self, val = 1):
        self.val = 1

    def sample(self):
        return self.val

    def mean(self):
        return self.val





class MultiClassMultiHopBP(EnvBase): # TODO: Make it compatible with torchrl EnvBase
    """
    Internal state is Q and Y (queues and link states)
    Q is size NxK where N is the number of nodes in the network and K is the number of classes
    Y is size Mx1 where M is the number of links in the network

    Actions are Mx(K+1) matrices where M is the number of links and K is the number of classes
    Action[m,k+1] denotes the number of packets to send on link m for class k
    Action[m,0] denotes the number of packets to idle on link m



    Matrix Representation:
        Y is a NxN matrix
        Q is a NxK matrix
        Action is a NxNx(K+1)
        Action[n,m,k+1] denotes the number of packets to send from node n to node m for class k

        Dynamics: Q[n,k] = max(Q[n,k] - sum(Action[n,:,k+1]), 0) + sum(Action[:,n,k+1])





    The reward is the negative of the sum of the elements of Q at time t, meaning before the action and new arrivals

    X(t) is sampled from a set of arrival distributions, which is a list of N+1 distributions (generators)
    Y(t) is sampled from a set of link state distributions, which is a list of N+1 distributions (generators)

    Notes:
        1. Assuming batch_size = 1 (i.e.number of environments each instance represents is 1)
        2.


    """



    batch_locked = False
    def __init__(self,
                 nodes: Union[List[int], int],
                 link_info: Dict,
                 class_info: Dict,
                 link_distribution: str = "fixed",
                 arrival_distribution: str = "poisson",
                 context_id: Optional[int] = 0,
                 max_backlog: Optional[int] = None,
                 device: Optional[str] = None,
                 action_func: Optional[str] = None,
                 **kwargs):
        super().__init__()

        # Set device -  where incoming and outgoing tensors lie
        self.device = device

        # Set Batch Size
        self.batch_size = torch.Size()

        # Interval ID
        self.context_id = torch.tensor(context_id, dtype = torch.float32)

        # Maximum backlog
        self.max_backlog = max_backlog

        # Internal random number generator
        self.rng = torch.Generator()

        # set action function
        if action_func == "bpi":
            self.action_func = self.backpressureWithInterference
        else:
            self.action_func = self.backpressure


        # Nodes/Buffers/Queues
        self.nodes = sorted(nodes) if isinstance(nodes, list) else list(range(nodes))
        self.num_nodes = self.N = len(self.nodes)

        # Initialize Links
        self.link_info = link_info
        self.num_links = self.M = len(link_info)
        self.link_distribution_type = link_distribution.lower()
        self.link_map = {}  #
        self.service_rates = torch.zeros(self.M)
        self.outgoing_links = OrderedDict() # for each node, a list of outgoing links
        self.incoming_links = OrderedDict()

        self._init_links(link_info)
        self.edge_index = torch.transpose(torch.tensor(list(self.link_map.values()), dtype = torch.long), 0, 1)
        self.link_rates = torch.tensor([link_dict["rate"] for link_dict in link_info], dtype = torch.float32)
        self.cap = self._sim_capacities() # Capacities at time $t=0$
        self.start_nodes = torch.tensor([self.link_map[m][0] for m in range(self.M)]) # Start node for each link
        self.end_nodes = torch.tensor([self.link_map[m][1] for m in range(self.M)])   # End node for each link

        # Initialize Classes
        self.class_info = class_info
        self.num_classes = self.K = len(class_info)
        self.Q = torch.zeros([self.N, self.K])
        self.arrival_map = {}  # Maps (start, end) to class id
        self.arrival_rates = torch.zeros(self.num_classes, dtype=torch.float32)
        self.destination_map = {}
        self.arrival_distribution_type = arrival_distribution.lower()

        self._init_classes(class_info)
        # self._process_arrivals() # Creates initial arrivals at $t=0$

        self.graphx = nx.DiGraph(list(self.link_map.values()))

        self.max_shortest_path = 100
        self.shortest_path_dist = {} # shortest path from node i to destination node of class k
        self.sp_dist = torch.zeros([self.N, self.K]) #Tensor form of shortest_path_dist


        self._init_shortest_path_dist()

        self.base_mask = torch.zeros([self.M, self.K], dtype = torch.bool)
        self._init_base_mask()

        # Create specs for torchrl
        self._make_spec()

    def _make_spec(self):
        """
        Ned to initialize the
        :return:
        """

        self.observation_spec = Composite({
            "Q": Unbounded(shape = self.Q.shape, dtype = torch.float32),
            "cap": Bounded(low = 0, high = 100, shape = self.cap.shape, dtype = torch.float32),
            "edge_index": Unbounded(shape = self.edge_index.shape, dtype = torch.long),
            "arrival_rates": Unbounded(shape = self.arrival_rates.shape, dtype = torch.float32),
            "link_rates": Unbounded(shape = self.link_rates.shape, dtype = torch.float32),
            "context_id": Unbounded(shape = self.context_id.shape, dtype = torch.float32),
            "mask": Binary(shape = (self.base_mask.shape[0], self.base_mask.shape[1]+1), dtype = torch.bool),
            "departures": Unbounded(shape = self.Q.shape, dtype = torch.float32),
            "arrivals": Unbounded(shape = torch.Size([self.K]), dtype = torch.float32),
            "sp_dist": Unbounded(shape = self.sp_dist.shape, dtype = torch.float32),


        }, batch_size=self.batch_size)
        self.action_spec = Bounded(low = 0, high = 1, shape = torch.Size([self.M, self.K+1]), dtype = torch.int)
        self.reward_spec = Unbounded(shape = torch.Size([1]), dtype = torch.float32)


    def _set_seed(self, seed):
        pass # all seeding should be done externally
    def _get_link_cap(self, start, end):
        return self.cap[self.link_map[(start, end)]]

    def _init_links(self, link_info):
        for id,link_dict in enumerate(link_info):
            self.link_map[id] = (link_dict["start"], link_dict["end"])
            self.service_rates[id] = link_dict["rate"]
            self.outgoing_links[link_dict["start"]] = self.outgoing_links.get(link_dict["start"], []) + [id]
            self.incoming_links[link_dict["end"]] = self.incoming_links.get(link_dict["end"], []) + [id]
        self._sim_capacities =  self._init_random_process(self.service_rates, self.link_distribution_type)

    def _process_arrivals(self):
        arrivals = self._sim_arrivals()
        for id, class_dict in enumerate(self.class_info):
            start = self.arrival_map[id]
            self.Q[start,id] += arrivals[id]
        return arrivals

    def _process_departures(self):
        departures = torch.zeros_like(self.Q)
        for id, dest in self.destination_map.items():
            departures[dest, id] = self.Q[dest,id].item()
            self.Q[dest,id] = 0
        return departures

    def _init_classes(self, class_info):

        for id, class_dict in enumerate(class_info):
            self.arrival_map[id] = class_dict["source"]
            self.destination_map[id] = class_dict["destination"]
            self.arrival_rates[id] = class_dict["rate"]
        self._sim_arrivals = self._init_random_process(self.arrival_rates, self.arrival_distribution_type)


    def _init_random_process(self, rates, dist):
        """
        kwargs should contain the distribution (distribution = "poisson) and any parameters
        :param kwargs:
        :return:
        """
        if not isinstance(rates,torch.Tensor):
            rates = torch.Tensor(rates)
        if dist == "poisson":
            return lambda: torch.poisson(rates, generator=self.rng)
        elif dist == "fixed":
            return lambda: rates
        else:
            raise ValueError("Distribution not supported")
    def _reset(self, tensordict = None, **kwargs):

        self.Q = torch.zeros_like(self.Q)
        return self._get_observation(reset = True)

    def _step(self, td:TensorDict):
        # [:,-K:] is used to ensure that the last K columns are used in-case action is K+1
        # *self.base_mask ensures that we don't try to send packets to nodes that cannot reach the destination
        action = self.action_func(self.Q, self.cap, td["mask"], self.M, self.K, self.link_info)

        if self.action_func == self.backpressureWithInterference:
            start_Q = self.Q.sum().item()
            to_transmit = torch.min(action, self.Q[self.start_nodes])
            self.Q.index_add(0, self.start_nodes, -to_transmit)
            self.Q.index_add(0, self.end_nodes, to_transmit)
            end_Q = self.Q.sum().item()
            if end_Q - start_Q > 0:
                raise ValueError(f"Backpressure with interference increased the total number of packets in the network by {diff}")
        else:
            start_Q = self.Q.clone()
            diffQ = torch.zeros_like(self.Q)
        # for m in torch.randperm(self.M):
            for m in range(self.M):
                # (checks)
                # (1) ensures that the sum of packets transmitted over a link cannot exceed the capacity of the link
                assert action[m].sum(dim = 0) <= self.cap[m]
                # (2) ensures that we don't try to transmit more packets than are available at the start node
                to_transmit = torch.min(action[m], start_Q[self.start_nodes[m]])
                start_Q[self.start_nodes[m]] -= to_transmit # needed for (2)
                # (3) updates the difference in packets at the start and end nodes
                diffQ[self.start_nodes[m]] -= to_transmit
                diffQ[self.end_nodes[m]] += to_transmit
            self.Q += diffQ

        return self._get_observation()

    def backpressure(self, Q: torch.Tensor, cap: torch.Tensor, mask: torch.Tensor, M: int, K: int, link_info: dict):
        """
        Runs the backpressure algorithm given the network

        Backpressure algorithm:
            For each link:
                1. Find the class which has the greatest difference between the queue length at the start and end nodes THAT IS NOT MASKED
                2. Send the largest class using the full link capacity for each link if there is a positive differential between start and end nodes
                3. If there is no positive differential, send no packets i.e. a_i[0,1:] = 0, a_i[0,0] = Y[i]

        :param net: MultiClassMultiHop object
        :param td: TensorDict object, should contain "mask" which is a torch.Tensor of shape [M, K]

        :return: action: torch.Tensor of shape [M, K] where M is the number of links and K is the number of classes
        """

        # send the largest class using the full link capacity for each link
        action = torch.zeros([M, K])
        for m in range(M):
            diff = (Q[link_info[m]["start"]] - Q[link_info[m]["end"]]) * mask[m][1:]  # mask out the classes that are not allowed to be sent
            max_class = torch.argmax(diff)
            if diff[max_class] > 0:
                action[m, max_class] = cap[m]
        return action

    def backpressureWithInterference(self, Q: torch.Tensor, cap: torch.Tensor, mask: torch.Tensor, M: int, K: int, link_info: dict):
        """
        Runs the backpressure algorithm given the network

        Backpressure algorithm:
            For each link:
                1. Find the class which has the greatest difference between the queue length at the start and end nodes THAT IS NOT MASKED
                2. Send the largest class using the full link capacity for each link if there is a positive differential between start and end nodes
                3. If there is no positive differential, send no packets i.e. a_i[0,1:] = 0, a_i[0,0] = Y[i]

        :param net: MultiClassMultiHop object
        :param td: TensorDict object, should contain "mask" which is a torch.Tensor of shape [M, K]

        :return: action: torch.Tensor of shape [M, K] where M is the number of links and K is the number of classes
        """

        # send the largest class using the full link capacity for each link
        action = torch.zeros([M, K])
        weights = torch.zeros([M,1])
        chosen_class = torch.zeros([M,1], dtype = torch.int)
        for m in range(M):
            pressure = (Q[link_info[m]["start"]] - Q[link_info[m]["end"]]) * mask[m][1:]  # mask out the classes that are not allowed to be sent
            chosen_class[m] = torch.argmax(pressure)
            weights[m] = pressure[chosen_class[m]]

        """
        Now we need to take the argmax over all links that share the same start node
        """
        for n in range(self.N):
            # get the links that start at node n
            links = self.outgoing_links[n]
            # get the classes that have the highest pressure
            argmax_weight = torch.argmax(weights[links])
            action[links[argmax_weight], chosen_class[links[argmax_weight]]] = cap[links[argmax_weight]]

        return action*mask[:,1:]

    """
    Old _step() that requires converting to a valid action
    ------------------------------
    def _step(self, td:TensorDict): 

        # Step 1: Get valid action in case td["action"] violates constraints
        if td.get("valid_action", False):
            action = td["action"]
        else:
            action = self._get_valid_action(td["action"])
        if action.shape[1] == self.K +1:
            action = action[:,1:]

        # Step 2: Apply action to queues
        
        Assuming the action is a valid (M,K) tensor where (m,k) denotes the number of packets to transmits over start 
        node of link m to end node of link m of class k
        
        for m in range(self.M):
            self.Q[self.start_nodes[m]] -= action[m]
            self.Q[self.end_nodes[m]] += action[m]

        return self._get_observation()
    """

    def convert_action(self, action):
        """
        Converts a (M,K) action tensor to a representation showing
        start_node: [end_node, action]
        :param action:
        :return:
        """
        if action.shape[1] == self.K + 1:
            action = action[:,1:]
        return {self.link_map[m]: action[m].tolist() for m in range(self.M)}





    def _get_observation(self, reset = False):
        # Step 3: Deliver all packets that have reached their destination
        departures = self._process_departures()

        # Step 4: Compute reward, with shape (1,)
        reward = -torch.sum(self.Q).reshape([1])

        # Step 5: Check if the episode should be truncated
        terminated = False
        if self.max_backlog is not None and -reward > self.max_backlog:
            # reward = -torch.Tensor(100)
            terminated = True

        # Step 6: Generate new arrivals
        arrivals = self._process_arrivals()

        # Step 7: Generate new link capacities
        self._sim_capacities()

        td = TensorDict({
            "Q": self.Q.clone(),
            "cap": self.cap,
            "departures": departures,
            "arrivals": arrivals,
            "mask": self._get_mask(),
            "sp_dist": self.sp_dist,
            "edge_index": self.edge_index,
            "link_rates": self.link_rates,
            "arrival_rates": self.arrival_rates,
            "context_id": self.context_id,
            "terminated": torch.tensor([terminated], dtype=bool),

        }, batch_size=self.batch_size)

        if not reset:
            td.set("reward", reward)
        return td


    def _get_valid_action(self, action: torch.Tensor):
        """
        WARNING: THIS METHOD TAKES A LONG TIME TO RUN SO CARE SHOULD BE TAKEN TO AVOID HAVING TO CALL IT

        action is an (M,K) tensor where (m,k) denotes the number of packets to transmits over start node of link m to end node of link m of class k
        return a valid action where (m, k) is not greater than the number of packets in the start node of link m
        and where the sum over K of (m,k) is not greater than the capacity of link m

        How do we not send ghost packets? I.e. if the sum of packets that action wants to transmit from Q[i,k] is greater than Q[i,k]
        then we should only send Q[i,k] packets. But we should also not send more packets than the capacity of the link

        We have two constraints.
        1. The sum of packets transmitted over a link cannot exceed the capacity of the link
        2. The sum of packets transmitted from a particular queue cannot exceed the number of packets in the queue

        To implement this:
        1. Get the minimum of the action and the number of packets in the start node -> valid_action
        2. For each Q[n,k] the sum over j in self.outgoing_links[n]of valid_action[j,k] cannot exceed Q[n,k] -> valid_action
        3. For each link m, the sum over k of valid_action[m,k] cannot exceed the capacity of the link -> valid_action

        :param action:
        :return:
        """
        if action.dim() == 1:
            # convert to (M,K+1) where
            # new_action[m,k] = cap[m] if action[m] = k
            new_action = torch.zeros([self.M, self.K+1])
            new_action[torch.arange(self.M), action] = self.cap
            action = new_action
        if action.shape[1] == self.K + 1:
            action = action[:,1:]
        action = self.base_mask* action
        valid_action = torch.zeros_like(action)
        # copy of capacity
        residual_capacity = self.cap.clone()
        # copy of Q
        residual_packets = self.Q.clone()
        # Loop through and randomize priority
        for m in torch.randperm(self.M):
            if residual_capacity[m] == 0: # speedup
                continue
            for k in torch.randperm(self.K): # speedup
                if residual_packets[self.start_nodes[m],k] == 0:
                    continue
                # valid_action[m,k] must be the minimum of:
                # 1. action[m,k] - the number of class k packets at link m to transmit
                # 2. residual_packets[self.start_nodes[m],k] - the number of class k packets at the start node of link m
                # 3. residual_capacity[m] - the remaining capacity of link m
                valid_action[m,k] = torch.min(action[m,k], torch.min(residual_packets[self.start_nodes[m],k], residual_capacity[m]))
                residual_packets[self.start_nodes[m],k] = torch.clamp(residual_packets[self.start_nodes[m],k]-valid_action[m,k], min = 0)
                residual_capacity[m] = torch.clamp(residual_capacity[m] - valid_action[m,k], min = 0)
        return valid_action


    def _get_random_valid_action(self):
        # samples a random action where each action[m,k] < cap[m]
        action = torch.ones([self.M, self.K])*self.cap.max()
        return self._get_valid_action(action)

    """
    Need to create a set of methods for:
    1. Checking if there is a path from node i to the destination node of class k
    2. If not, updating the mask such that all actions that try to send class k packets from node i 
        to a node that is not connected to the destination node of class k are set to zero
    """

    def _init_shortest_path_dist(self):
        """
        Get the shortest_path from each node to the destination node of each class
        if there is not a shortest path  node i to destination k
        then set the shortest_path_length to a very large number

        :return:
        """

        for source, dest_dict in nx.all_pairs_shortest_path_length(self.graphx):
            self.shortest_path_dist[source] = {}
            for n in range(self.N):
                if n in dest_dict:
                    self.shortest_path_dist[source][n] = dest_dict[n]
                else:
                    self.shortest_path_dist[source][n] = self.max_shortest_path
        # Convert to tensor
        for n in range(self.N):
            for k in range(self.K):
                self.sp_dist[n,k] = self.shortest_path_dist[n][self.destination_map[k]]


    def _init_base_mask(self):
        """
        Creates an (M,K) base_mask where base_mask[m,k] is true if the end node
        of link m can reach the destination of class k
        We can check this by seeing if self.shortest_path_dist[self.end_nodes[m]][self.destination_map[k]] < self.max_shortest_path]
        :return:
        """
        for m in range(self.M):
            for k in range(self.K):
                self.base_mask[m,k] = self.shortest_path_dist[self.end_nodes[m].item()][self.destination_map[k]] < self.max_shortest_path

    def _get_mask(self):
        """
        In sets mask (m,k) to False if the start node of link m has no packets of class k
        :return:
        """
        mask = self.Q[self.start_nodes] > 0 # Prevents sending packets from nodes with no packets
        mask = torch.logical_and(mask, self.base_mask) # Prevents sending packets to nodes that cannot reach the destination
        any_true = torch.any(mask, dim = 1,keepdim = True) # Allows idling only if there are no other valid actions
        return torch.concat([~any_true, mask], dim = 1)
        # ones = torch.ones([mask.shape[0],1], dtype = torch.bool)
        # return torch.concat([ones,torch.logical_and(mask, self.base_mask)],dim=1)




