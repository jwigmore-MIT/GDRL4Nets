from torch_geometric.nn import SAGEConv


from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt
from torchrl.envs.transforms import ObservationTransform, TransformedEnv
from modules.torchrl_development.envs.custom_transforms import MCMHPygTransform
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
from tensordict import (
    TensorDictBase,
)
from torchrl.data.tensor_specs import (
    ContinuousBox,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    Unbounded,
    Composite,
    NonTensor,
)


file_path = "../envs/env2.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

# init env
env = MultiClassMultiHop(**env_info)
env = TransformedEnv(env, MCMHPygTransform(in_keys=["Q"], out_keys=["X"], env=env))

# Get rollout for data
td = env.rollout(max_steps = 20)

# Create single SageConv Layer
layer0 = SAGEConv(in_channels=env.observation_spec["X"].shape[-1],
                  out_channels=32,
                  aggr = "mean",
                  normalize=False)

# Get Data in correct format
X = td["X"][0]
edge_index = td["edge_index"][0]

# Forward pass
Xprime = layer0(X, edge_index)
"""
This is equivalent to $W_s X+M_{v^-}$

To not include W_s X we can do:
"""
layer0 = SAGEConv(in_channels=env.observation_spec["X"].shape[-1],
                    out_channels=32,
                    aggr = "mean",
                    normalize=False,
                    root_weight=False)

Xprime = layer0(X, edge_index)

"""
If we want to include reversible connections we can do:

This produces $\mathcal M_{v^-} + \mathcal M_v^+$
"""
from torch_geometric.nn import DirGNNConv

node_message_layer = DirGNNConv(
    conv = SAGEConv(in_channels=env.observation_spec["X"].shape[-1],
                    out_channels=32,
                    aggr = "mean",
                    normalize=False,
                    root_weight=False),
    alpha = 0.5, # equal weighting for forward and reverse neighbors
    root_weight=False,)

node_message_0 = layer0(X, edge_index)

"""
Next we want to get the class level message
"""

class_message_layer = SAGEConv(
                        in_channels=env.observation_spec["X"].shape[-1],
                        out_channels=32,
                        aggr = "mean",
                        normalize=False,
                        root_weight=False)


class_message_0 = class_message_layer(X, td["class_edge_index"][0])




"""
Currently graph sage does not support edge features. We can add edge features by using the GATConv layer.
Or we can ignore edge features and assume that all edges have unit capacity and are always active.  

Might be able to use pyg.

"""

# Create a class that combines the node and class message layers
from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
from torch import Tensor

from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
class MCHCNodeOnlySageConv(MessagePassing):


    def __init__(self,
                 in_channels = 32,
                 out_channels = 32,
                 aggr = "mean",
                 normalize = False,
                 activation = F.relu,
                 **kwargs
                 ):

        super(MCHCNodeOnlySageConv, self).__init__(aggr = aggr, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.node_message_layer = DirGNNConv(conv = SAGEConv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    aggr = aggr,
                    normalize=self.normalize,
                    root_weight=False),
                    alpha = 0.5, # equal weighting for forward and reverse neighbors
                    root_weight=False,)

        self.class_message_layer = SAGEConv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    aggr = aggr,
                    normalize=self.normalize,
                    root_weight=False)

        self.self_layer = Linear(self.in_channels, self.out_channels, bias = True)

        self.activation = activation

    def forward(self, x: Tensor, edge_index: Adj, class_edge_index: Adj) -> Tensor:

        # Get node message
        node_message = self.node_message_layer(x, edge_index)

        # Get class message
        class_message = self.class_message_layer(x, class_edge_index)

        # Get self message
        self_message = self.self_layer(x)

        return self.activation(node_message + class_message + self_message)



class MCMHEdgeDecoder(torch.nn.Module):
    """
    Take the node embeddings and for each edge (i,j) computes an edge embedding:
    E[i,j] = W_{e1}X[i] + W_{e2}X[j]
    """
    def __init__(self, in_channels, out_channels=1):
        super(MCMHEdgeDecoder, self).__init__()
        self.W1 = Linear(in_channels, out_channels, bias = False)
        self.W2 = Linear(in_channels, out_channels, bias = False)

    def forward(self, X, edge_index):
        return self.W1(X[edge_index[0]]) + self.W2(X[edge_index[1]])

# Test the MCHCNodeOnlySageConv layer



node_only_sage = MCHCNodeOnlySageConv(in_channels = td["X"].shape[-1],
                                        out_channels = 32,
                                        aggr = "mean",
                                        normalize = False,
                                        activation = F.relu)



Xprime = node_only_sage(td["X"][0], td["edge_index"][0], td["class_edge_index"][0])


# Create a Multilayer GNN from the MCHCNodeOnlySageConv layer
from torch_geometric.nn.models.basic_gnn import BasicGNN
class MCHCGraphSage(torch.nn.Module):

    """
    Modeled after  BasicGNN... somewhat

    After the final layer, we will have node embeddings for each node-class, we then want to compute edge embeddings


    """
    def __init__(self,
                 in_channels: int,
                 hidden_channels: int,
                 num_layers: int,
                 aggregation: str = "mean",
                 normalize: bool = False,
                 ):
        super(MCHCGraphSage, self).__init__()
        self.node_layers = torch.nn.ModuleList()
        self.node_layers.append(MCHCNodeOnlySageConv(in_channels = in_channels,
                                                out_channels = hidden_channels,
                                                aggr = aggregation,
                                                normalize = normalize,
                                                activation = F.relu))
        for i in range(1,num_layers):
            self.node_layers.append(MCHCNodeOnlySageConv(in_channels = hidden_channels,
                                                     out_channels = hidden_channels,
                                                     aggr = aggregation,
                                                     normalize = normalize,
                                                     activation = F.relu,
                                                     ))
        self.edge_decoder = MCMHEdgeDecoder(in_channels = hidden_channels, out_channels = 1)


    def forward(self, x, edge_index, class_edge_index):
        for layer in self.node_layers:
            x = layer(x, edge_index, class_edge_index)
        return self.edge_decoder(x, edge_index)

# Test the MCHCGraphSage

graph_sage = MCHCGraphSage(in_channels = td["X"].shape[-1],
                            hidden_channels = 32,
                            num_layers = 2,
                            aggregation = "mean",
                            normalize = False)

edge_embeddings = graph_sage(td["X"][0], td["edge_index"][0], td["class_edge_index"][0])

edge_class_embeddings = edge_embeddings.view(env.N, -1)

# Perform softmax on each row of edge_class_embeddings
edge_class_probs = F.softmax(edge_class_embeddings, dim = -1)

# Test MCHCGraphSage on a batch of data

batch_graph = tensors_to_batch(td["X"], td["edge_index"], td["class_edge_index"], M = env.M)
# Batch graph.x is a tensor of shape (batch_size*N*K, node_features)

edge_embeddings = graph_sage(batch_graph.x, batch_graph.edge_index, batch_graph.class_edge_index)
# edge embeddings will be (batch_size*M*K,1)

#How can I recover M from edge_embeddings without knowing env.M?


edge_class_embeddings = edge_embeddings.view(batch_graph.batch_size,batch_graph.M , -1)

# Perform softmax on each row of edge_class_embeddings
edge_class_probs = F.softmax(edge_class_embeddings, dim = -1)


