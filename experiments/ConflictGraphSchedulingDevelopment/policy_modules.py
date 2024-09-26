import torch.nn as nn
from torch_geometric.nn import DeepSetsAggregation, DeepGCNLayer, GENConv
from modules.torchrl_development.nn_modules.neighbor_argmax import NeighborArgmax
import torch.nn.functional as F
from torch.nn import LayerNorm
from torch.nn import Linear, ReLU, Sigmoid, LeakyReLU
import torch
from torch_geometric.nn import Sequential, GCNConv, SAGEConv, GCN

class GCN_Policy_Module(nn.Module):

    def __init__(self, node_features, num_layers):
        super().__init__()
        self.main_module = GCN(in_channels = node_features,
                               hidden_channels = 32,
                               out_channels = 1,
                               num_layers = num_layers,
                               act = LeakyReLU,
                               )

        self.sigmoid = Sigmoid()
        self.argmax = NeighborArgmax(in_channels = -1)

    def forward(self, x, edge_index, batch = None):
        logits = self.main_module(x, edge_index)
        if self.training:
            x = self.sigmoid(logits)
        else:
            x = self.argmax(logits, edge_index)
        return x, logits
class Policy_Module2(nn.Module):
    class DeeperGCN(nn.Module):
        def __init__(self, input_size,  hidden_channels, num_layers, block = "res+"):
            super().__init__()

            self.node_encoder = Linear(input_size, hidden_channels)

            self.layers = torch.nn.ModuleList()
            for i in range(1, num_layers + 1):
                conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                               t=1.0, learn_t=True, num_layers=2, norm='layer')
                norm = LayerNorm(hidden_channels, elementwise_affine=True)
                act = ReLU(inplace=True)

                layer = DeepGCNLayer(conv, norm, act, block=block, dropout=0.1,
                                     ckpt_grad=i % 3)
                self.layers.append(layer)

            self.lin = Linear(hidden_channels, 1)

        def forward(self, x, edge_index):
            x = self.node_encoder(x)

            x = self.layers[0].conv(x, edge_index)

            for layer in self.layers[1:]:
                x = layer(x, edge_index)

            x = self.layers[0].act(self.layers[0].norm(x))
            x = F.dropout(x, p=0.1, training=self.training)

            return self.lin(x)

    def __init__(self, node_features, hidden_channels = 64, num_layers = 2, block = "res+"):
        super().__init__()
        self.main_module = self.DeeperGCN(node_features, hidden_channels, num_layers, block = block)
        self.sigmoid = Sigmoid()
        self.argmax = NeighborArgmax(in_channels = -1)

    def forward(self, x, edge_index, batch = None):
        logits = self.main_module(x, edge_index)
        if self.training:
            x = self.sigmoid(logits)
        else:
            x = self.argmax(logits, edge_index)
        return x, logits
class Policy_Module(nn.Module):
    def __init__(self, node_features):
        super(Policy_Module, self).__init__()
        # self.conv1 = SAGEConv(node_features, 32, project = False, aggr = DeepSetsAggregation())
        self.conv1 = DeepGCNLayer(GENConv(node_features, 64, aggr = "softmax", t=1.0),
                                 block = "dense")
        # self.conv2 = SAGEConv(32, 32, project = False, aggr = DeepSetsAggregation())
        self.conv2 = SAGEConv(64+node_features, 64, project = True, aggr = DeepSetsAggregation())
        # self.relu = ReLU()
        self.relu = Sigmoid()
        self.fc = Linear(64, 1)
        self.sigmoid = Sigmoid()
        self.argmax = NeighborArgmax(in_channels = -1)

    def forward(self, x, edge_index, batch = None):
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        logits = self.fc(x)
        if self.training:
            x = self.sigmoid(logits)
        else:
            x = self.argmax(logits, edge_index)
        return x, logits