# import
from copy import deepcopy
from typing import Optional, List, Union
from collections import OrderedDict

import numpy as np
import torch
from tensordict import TensorDict
ListLike = Union[List, np.ndarray, torch.Tensor]

from torchrl.data import BoundedTensorSpec, CompositeSpec, Bounded, Unbounded, Binary
from torchrl.envs import (
    EnvBase,
)
from torch_geometric.utils import dense_to_sparse

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


""" Overview:
This is an environment for a conflict graph scheduling problem.
The input is:
 1. an adjacency matrix of a undirected graph, where each node represents a wireless link, and each edge represents a conflict between two links.
 2. Arrival distribution of packets for each node (Poisson, Bernoulli)
 3. Arrival rate for each node
 4. Service distribution for each node (Poisson, Bernoulli, Fixed)
 5. Service rate for each node
 
 There are a total of N nodes and M edges in the graph. Nodes are conflicting if they are connected by an edge.  The goal
 is to schedule packets such that no two conflicting nodes are scheduled at the same time.
 This is an independent set problem.
 
 If two nodes are conflicting, and they are both scheduled at the same time, then the transmission is unsuccessful, and 
 the queue sizes of the conflict nodes don't decrease.  
 
 The environment is a discrete time system where at each time step, the following happens:
 0. The network is in state (q, y), where q is the queue size of each node, and y is the service state of each node.
 1. an action is input into the system. This action 'a' is represented as a binary tensor of size (B, N), where B is the 
 batch size and N is the number of nodes.
 2. The action is checked to see if it is valid. An action is valid if no two conflicting nodes are selected. If the 
 action is invalid, then the action is modified to be valid.
 3. The system then applies the valid schedule to the system and arrivals x(t) are added. The queues evolve as follows:
    q(t+1) = q(t)- y(t)*a'(t) +  x(t)
    where a'(t) is the valid action
 4. The service states are updated based on their service distribution
 
    The reward is the sum of the queue sizes at each node. The goal is to minimize the sum of the queue sizes.
 
"""





class ConflictGraphScheduling(EnvBase):

    def __init__(self,
                 adj: ListLike,
                 arrival_dist: str,
                 arrival_rate: ListLike,
                 service_dist: str,
                 service_rate: ListLike,
                 seed: Optional[int] = None,
                 batch_size: int = 1,
                 max_queue_size: Optional[int] = None,
                 node_priority: str = None,
                 context_id: Optional[int] = -1,
                 interference_penalty: Optional[float] = 0.0,
                 reset_penalty: Optional[float] = 0.0,
                 **kwargs):

        super().__init__(batch_size = ())
        self.context_id = torch.Tensor([context_id])
        self.interference_penalty = interference_penalty
        self.reset_penalty = reset_penalty
        self.adj = torch.Tensor(adj)
        self.adj_sparse = dense_to_sparse(self.adj)[0]
        self.num_nodes = self.adj.shape[0]
        self.num_edges = int((torch.sum(self.adj) // 2).item())

        # intialize queue and service states
        self.q = torch.zeros(self.num_nodes)
        self.s = torch.zeros(self.num_nodes)

        # Create node priority
        if node_priority is None:
            self.node_priority = torch.zeros(self.num_nodes).float()
        elif node_priority == "random":
            self.node_priority = torch.randperm(self.num_nodes).float()
        elif node_priority == "increasing":
            self.node_priority = torch.arange(self.num_nodes).float()
        else:
            raise ValueError("node_priority must be one of None, 'random', 'increasing'")

        # create torch generator object
        self.rng = torch.Generator()
        self.arrival_dist = arrival_dist
        self.arrival_rate = torch.Tensor(arrival_rate)
        self.service_dist = service_dist
        self.service_rate = torch.Tensor(service_rate)
        self.seed = seed
        self.set_seed(seed)

        # initialize internal random processes
        self._init_random_processes()

        # set maximum q size
        self.max_queue_size = max_queue_size

        # create specs
        self._make_specs()

    def _init_random_processes(self):
        if self.arrival_dist == "Poisson":
            self._sim_arrivals = lambda: torch.poisson(self.arrival_rate, generator = self.rng)
        elif self.arrival_dist == "Bernoulli":
            self._sim_arrivals = lambda: torch.bernoulli(self.arrival_rate, generator = self.rng)
        elif self.arrival_dist == "Fixed":
            self._sim_arrivals = lambda: torch.Tensor(self.arrival_rate)
        else:
            raise ValueError("arrival_dist must be one of 'Poisson', 'Bernoulli', 'Fixed'")

        if self.service_dist == "Poisson":
            self._sim_services = lambda: torch.poisson(self.service_rate, generator = self.rng)
        elif self.service_dist == "Bernoulli":
            self._sim_services = lambda: torch.bernoulli(self.service_rate, generator = self.rng)
        elif self.service_dist == "Fixed":
            self._sim_services = lambda: torch.Tensor(self.service_rate)
        elif self.service_dist == "Normal":
            self._sim_services = lambda: torch.normal(self.service_rate[0], self.service_rate[1], generator=self.rng).clamp(min =0, max = 100).round()
        else:
            raise ValueError("service_dist must be one of 'Poisson', 'Bernoulli', 'Fixed', 'Normal'")

    def _step(self, td: TensorDict):


        # Get valid action
        action = self._get_valid_action(td["action"])

        # get arrivals
        arrivals = self._sim_arrivals()

        # apply valid action
        self.q = torch.clamp(self.q - self.s * action + arrivals, min = 0)

        # update service states
        self.s = self._sim_services()

        # calculate reward
        reward =  -torch.sum(self.q) - self.interference_penalty*(td["action"]-action).sum()

        # check if terminated
        if self.max_queue_size is not None:
            terminated = torch.Tensor([reward < -self.max_queue_size]).bool()
            reward -= self.reset_penalty
        else:
            terminated = torch.Tensor([False]).bool()

        out = TensorDict({
            "q": self.q,
            "s": self.s,
            "valid_action": action,
            "adj_sparse": self.adj_sparse,
            "reward": reward,
            "terminated": terminated,
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "context_id": self.context_id,
            "node_priority": self.node_priority,
        }, td.shape)
        return out

    def _reset(self, td: TensorDict = None):

        self.q = torch.zeros(self.num_nodes)
        self.s = self._sim_services()

        out = TensorDict({
            "q": self.q,
            "s": self.s,
            "valid_action": torch.zeros_like(self.q),
            "adj_sparse": self.adj_sparse,
            "terminated": torch.Tensor([False]).bool(),
            "reward": torch.Tensor([0.0]),
            "arrival_rate": self.arrival_rate,
            "service_rate": self.service_rate,
            "context_id": self.context_id,
            "node_priority": self.node_priority
        },

            self.batch_size)

        return out

    def _set_seed(self, seed: Optional[int]):
        if seed is not None:
            self.rng.manual_seed(int(seed))


    def _make_specs(self):
        """
        Create all the required specs for torchrl environments
        :return:
        """
        self.observation_spec = CompositeSpec({
            "q": Unbounded(shape = (self.num_nodes,), dtype = torch.float32),
            "s": Bounded(low = 0, high = 100, shape = (self.num_nodes,), dtype = torch.float32),
            "valid_action": Binary(n = self.num_nodes, shape = (self.num_nodes,), dtype = torch.float32),
            "adj_sparse": Unbounded(shape = self.adj_sparse.shape, dtype = torch.int64),
            "reward": Unbounded(shape = (1,), dtype = torch.float32),
            "arrival_rate": Unbounded(shape = (self.num_nodes,), dtype = torch.float32),
            "service_rate": Unbounded(shape = self.service_rate.shape, dtype = torch.float32),
            "context_id": Unbounded(shape = (1,), dtype = torch.float32),
            "node_priority": Unbounded(shape = (self.num_nodes,), dtype = torch.float32),
        })

        self.action_spec = Binary(n = self.num_nodes, shape = (self.num_nodes,), dtype = torch.float32)

        self.reward_spec = CompositeSpec({"reward": Bounded(low = -100_000, high = 0, shape = (1,), dtype = torch.float32)})



    def _get_valid_action(self, action: torch.Tensor):
        """
        This method converts a potentially invalid action to a valid action. An action is considered invalid if does not correspond to an independent set in the graph.
        The method checks the dimension of the input tensor and reshapes it if necessary.
        It returns an independent set (or a zero set) that is a subset of the input action, such that no two conflicting nodes are selected.

        Parameters:
        action (torch.Tensor): A tensor of shape (B, N) or (N,), where B is the batch size and N is the number of nodes in the graph.

        Returns:
        torch.Tensor: A tensor of the same shape as the input, but with invalid actions filtered out. Each element in the tensor is either 0 (invalid action) or 1 (valid action).
        """
        # Check the dimension of the input tensor
        if action.ndim == 1:
            # If the input tensor is 1D, add an extra dimension to make it 2D
            action = action.unsqueeze(0)
        # make any element zero if q is less than 1
        action = action * (self.q > 0).unsqueeze(0)

        # Transpose the action tensor to match the shape of the adjacency matrix
        action_transpose = torch.transpose(action, 0, 1)

        # Perform a series of operations involving the adjacency matrix and the transposed action tensor
        # The result is a tensor where each element indicates whether the corresponding action is valid or not
        return torch.multiply(torch.multiply(self.adj.unsqueeze(dim=-1), action_transpose).sum(axis=-2) < 1,
                              action_transpose).T.squeeze()

def compute_valid_actions(env: ConflictGraphScheduling) -> torch.Tensor:
    """
    Computes all valid actions for the given environment
    :param env:
    :return valid_actions: a tensor of shape (O(2^N), N) where each row is a valid action
    """
    N = env.num_nodes
    env.base_env.q = torch.ones(N)
    valid_actions = []
    for i in range(2**N):
        action = torch.Tensor([int(x) for x in list(bin(i)[2:].zfill(N))])
        valid_action = env._get_valid_action(action)
        if torch.all(valid_action == action):
            valid_actions.append(action)
    return torch.stack(valid_actions, dim = 0)

def maxweight(valid_actions, q, s):
    """
    Compute the maxweight scheduling policy
    argmax(q*y*a) where a is a valid action
    :param valid_actions: (K, N) binary tensor where K is the number of valid actions
    :param q: (B,N) tensor of queue sizes
    :param s: (B, N) tensor of service states
    :return: (B, N) tensor of the maxweight action
    """
    if q.ndim == 1:
        q = q.unsqueeze(0)
        s = s.unsqueeze(0)

    return torch.argmax(q.unsqueeze(1)*s.unsqueeze(1)*valid_actions, dim = 0)

if __name__ == "__main__":
    from torchrl.envs.utils import check_env_specs
    adj = np.array([[0,1,1,1,0], [1,0,1,1,0], [1,1,0,0,1], [1,1,0,0,1], [0,0,1,1,0]])
    arrival_dist = "Bernoulli"
    arrival_rate = np.array([0.1, 0.2, 0.3, 0.1, 0.1])
    service_dist = "Fixed"
    service_rate = np.array([1, 1, 1, 1,1])
    max_queue_size = 100

    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate, max_queue_size = max_queue_size)
    check_env_specs(env)
    td = env.reset()
    print(td)

    # Compute all valid actions for the environment
    valid_actions = compute_valid_actions(env)



