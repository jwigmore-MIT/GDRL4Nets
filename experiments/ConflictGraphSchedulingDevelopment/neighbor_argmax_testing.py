from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor
import numpy as np
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm

from modules.torchrl_development.envs.env_creation import make_env_cgs
from modules.torchrl_development.agents.cgs_agents import tensors_to_batch, GNN_TensorDictModule, IndependentBernoulli
import torch
from torchrl.envs import ExplorationType, set_exploration_type

from modules.torchrl_development.nn_modules.neighbor_argmax import NeighborArgmax

from torch_geometric.nn import Sequential, GCNConv, SAGEConv
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import DeepSetsAggregation
from torchrl.modules import MLP, ProbabilisticActor




adj = np.array([[0,1,0,0,], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
arrival_dist = "Bernoulli"
arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
service_dist = "Fixed"
service_rate = np.array([1, 1, 1, 1])

env_params = {"adj": adj, "arrival_rate": arrival_rate, "service_rate": service_rate, "arrival_dist": "Bernoulli", "service_dist": "Fixed"}
make_env_keywords = {"observation_keys": ["q", "s", "node_priority"], "stack_observation": True, "pyg_observation": False}

env = make_env_cgs(env_params, **make_env_keywords)
#
# q = torch.Tensor([[1,2,3,4]]).T
# adj_sparse = env.adj_sparse
#
# a_max_layer = NeighborArgmax(in_channels = 1)
#
# out = a_max_layer(q, adj_sparse)

# Collect rollout from environment
# td = env.rollout(max_steps = 100)
#
# batch_graph = tensors_to_batch(td["observation"], td["adj_sparse"])
# a_max_layer2 = NeighborArgmax(in_channels = -1)
# out2 = a_max_layer2(batch_graph.x, batch_graph.edge_index)


node_features = env.observation_spec["observation"].shape[-1]
policy_module = Sequential('x, edge_index, batch', [
            (SAGEConv(node_features, 16, project = True, aggr = 'mean'), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (SAGEConv(16, 16, project = True, aggr = 'mean'), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(16, 1),
            Sigmoid(),
            (NeighborArgmax(in_channels = -1), 'x, edge_index -> x')
        ])

actor = GNN_TensorDictModule(module = policy_module, x_key = "observation", edge_index_key = "adj_sparse", out_key = "probs")

actor = ProbabilisticActor(
    actor,
    in_keys=["probs"],
    distribution_class=IndependentBernoulli,
    spec = env.action_spec,
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM
    )


td = env.rollout(max_steps = 100, policy = actor)

actor.eval()

td2 = env.rollout(max_steps = 100, policy = actor)

