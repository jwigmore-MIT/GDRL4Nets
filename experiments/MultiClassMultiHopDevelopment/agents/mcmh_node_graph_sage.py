import torch.nn.functional as F
from torch import Tensor
import torch
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn import DirGNNConv
from torch_geometric.nn import SAGEConv
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
        # self.node_message_layer = DirGNNConv(conv = SAGEConv(
        #             in_channels=self.in_channels,
        #             out_channels=self.out_channels,
        #             aggr = aggr,
        #             normalize=self.normalize,
        #             root_weight=False),
        #             alpha = 0.5, # equal weighting for forward and reverse neighbors
        #             root_weight=False,)

        self.node_message_layer = SAGEConv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    aggr = aggr,
                    normalize=self.normalize,
                    root_weight=False)


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
        # class_message = self.class_message_layer(x, class_edge_index)

        # Get self message
        self_message = self.self_layer(x)

        # return self.activation(node_message + class_message + self_message)
        return self.activation(node_message + self_message)



class MCMHEdgeEncoder(torch.nn.Module):
    """
    Take the node embeddings and for each edge (i,j) computes an edge embedding:
    E[i,j] = W_{e1}X[i] + W_{e2}X[j]
    """
    def __init__(self, in_channels, out_channels=1):
        super(MCMHEdgeEncoder, self).__init__()
        self.W1 = Linear(in_channels, out_channels, bias = False)
        self.W2 = Linear(in_channels, out_channels, bias = False)

    def forward(self, X, edge_index):
        return self.W1(X[edge_index[0]]) + self.W2(X[edge_index[1]])


class MCMHEdgeDecoder(torch.nn.Module):

    def __init__(self, in_channels, out_channels=1):
        super(MCMHEdgeDecoder, self).__init__()
        self.layer = SAGEConv(in_channels, out_channels, aggr = "max", normalize = False, root_weight = True)

    def forward(self, X, edge_index):
        return self.layer(X, edge_index)


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
        self.edge_encoder = MCMHEdgeEncoder(in_channels = hidden_channels, out_channels = hidden_channels)
        self.edge_decoder = MCMHEdgeDecoder(in_channels = hidden_channels, out_channels = 1)




    def forward(self, x, edge_index, class_edge_index, physical_edge_index):
        for layer in self.node_layers:
            x = layer(x, edge_index, class_edge_index)
        edge_embeddings = self.edge_encoder(x, edge_index)
        return self.edge_decoder(edge_embeddings, physical_edge_index)

