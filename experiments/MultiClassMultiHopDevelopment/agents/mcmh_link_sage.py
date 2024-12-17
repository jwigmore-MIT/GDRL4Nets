import torch.nn.functional as F
from torch import Tensor
import torch
from torch_geometric.nn.aggr import Aggregation, MultiAggregation
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.aggr import SoftmaxAggregation
from torch_geometric.typing import Adj, OptPairTensor, Size, SparseTensor
from torch_geometric.utils import spmm
from torch_geometric.nn import DirGNNConv
from torch_geometric.nn import SAGEConv
from tensordict.nn import (
    TensorDictModule,
)
from tensordict import TensorDict
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
from torch_geometric.data import Batch
from torch_geometric.nn.pool import global_add_pool
class GNN_Actor(TensorDictModule):

    def __init__(self,
                 module,
                 feature_key="X",
                 edge_index_key = "edge_index",
                 class_edge_index_key = "class_edge_index",
                 out_keys = ["logits", "probs"],
                 small_logits = torch.Tensor([-1.0]),
                 valid_action = False):
        super(GNN_Actor, self).__init__(module = module, in_keys=[feature_key, edge_index_key, class_edge_index_key], out_keys=out_keys)

        self.feature_key = feature_key
        self.edge_index_key = edge_index_key
        self.class_edge_index_key = class_edge_index_key
        self.small_logits = small_logits
        self.valid_action = valid_action
        if self.valid_action:
            raise NotImplementedError("Valid action functionality does not work as its possible to transmit "
                                      "more packets over multiple links than exist in shared start node")

    def forward(self, input):
        if isinstance(input, TensorDict) and isinstance(input["X"], Batch): # Probabilistic actor automatically converts input to a TensorDict
            input = input["X"]
        if isinstance(input, TensorDict):
            K = input["Q"].shape[-1]
            if input[self.feature_key].dim() < 3: # < 3 # batch size is 1, meaning $\tilde X$ has shape [NK,F] an
                logits = self.module(input[self.feature_key],
                                     input[self.edge_index_key],
                                     input[self.class_edge_index_key]
                                     )
                logits = logits.reshape(K, -1).T
                logits = torch.cat((self.small_logits.repeat(logits.shape[0], 1), logits), dim=1)
                probs = torch.softmax(logits, dim=-1)
                input[self.out_keys[0]] = logits.squeeze(-1)
                input[self.out_keys[1]] = probs.squeeze(-1)
                if self.valid_action:

                    input["valid_action"] = torch.Tensor([1]).bool()
            else:
                batch_graph = tensors_to_batch(input[self.feature_key], input[self.edge_index_key], input[self.class_edge_index_key], K = K)
                logits = self.module(batch_graph.x, batch_graph.edge_index, batch_graph.class_edge_index)
                logits = logits.reshape(batch_graph.batch_size, K, -1).transpose(1, 2)
                input[self.out_keys[0]] = torch.cat((self.small_logits.expand(logits.shape[0], logits.shape[1], 1), logits), dim=-1)
                input[self.out_keys[1]] = torch.softmax(input[self.out_keys[0]], dim=-1)
                # if self.valid_action: # this will only be done during the update and not rollout stage
                #     input["valid_action"] = torch.ones_like(input[self.out_keys[0]][:, 0]).bool()
            return input
        elif isinstance(input, Batch):
            logits = self.module(input.x, input.edge_index)
            logits = logits.reshape(input.batch_size, input.K,-1).transpose(1,2)
            logits = torch.cat((self.small_logits.expand(logits.shape[0],logits.shape[1],1), logits), dim = -1)
            # probs = torch.softmax(logits, dim = -1)
            return logits  #, probs

class GNN_Critic(TensorDictModule):

    def __init__(self,
                 module,
                 feature_key="X",
                 edge_index_key = "edge_index",
                 class_edge_index_key = "class_edge_index",
                 out_keys = ["state_value"]):
        super(GNN_Critic, self).__init__(module = module, in_keys=[feature_key, edge_index_key, class_edge_index_key], out_keys=out_keys)

        self.feature_key = feature_key
        self.edge_index_key = edge_index_key
        self.class_edge_index_key = class_edge_index_key

    def forward(self, input):
        if isinstance(input, TensorDict) and isinstance(input["X"], Batch): # Probabilistic actor automatically converts input to a TensorDict
            input = input["X"]
        if isinstance(input, TensorDict):
            K = input["Q"].shape[-1]
            if input[self.feature_key].dim() < 3: # < 3 # batch size is 1, meaning $\tilde X$ has shape [NK,F] an
                logits = self.module(input[self.feature_key],
                                     input[self.edge_index_key],
                                     input[self.class_edge_index_key]
                                     )
                logits = logits.reshape(2, -1).T
                input[self.out_keys[0]] = global_add_pool(logits, None)
            else:
                batch_graph = tensors_to_batch(input[self.feature_key], input[self.edge_index_key], input[self.class_edge_index_key], K = K)
                logits = self.module(batch_graph.x, batch_graph.edge_index, batch_graph.class_edge_index)
                input[self.out_keys[0]] = global_add_pool(logits, batch_graph.batch)
            return input
        elif isinstance(input, Batch):
            logits = self.module(input.x, input.edge_index, input.class_edge_index)
            state_value = global_add_pool(logits, input.batch)
            return state_value  #, probs


class MCHCLinkSageConv(MessagePassing):


    def __init__(self,
                 in_channels = 32,
                 out_channels = 32,
                 aggregation = "mean",
                 normalize = False,
                 activation = F.leaky_relu,
                 project = False,
                 **kwargs
                 ):
        if aggregation == "softmax":
            aggregation = SoftmaxAggregation(learn=True)


        super(MCHCLinkSageConv, self).__init__(aggr = aggregation, **kwargs)

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
                    aggr = aggregation,
                    normalize=self.normalize,
                    root_weight=False,
                    project = project)


        self.class_message_layer = SAGEConv(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    aggr = aggregation,
                    normalize=self.normalize,
                    root_weight=False,
                    project = project)

        self.self_layer = Linear(self.in_channels, self.out_channels, bias = True)

        self.activation = activation

    def forward(self, x: Tensor, edge_index: Adj, class_edge_index: Adj) -> Tensor:

        # Get node message
        node_message = self.node_message_layer(x, edge_index)

        # Get class message
        class_message = self.class_message_layer(x, class_edge_index)

        # Get self message
        self_message = self.self_layer(x)

        # return self.activation(node_message + class_message + self_message)
        return self.activation(node_message + self_message)

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
                 activate_last_layer = True,
                 project_first = False,
                 ):
        super(MCHCGraphSage, self).__init__()
        self.node_layers = torch.nn.ModuleList()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        for i in range(0,num_layers):
            if i == 0:
                input_channels = self.in_channels
                if project_first:
                    project = True
                else:
                    project = False
            else:
                input_channels = hidden_channels
            if i == num_layers-1:
                out_channels = 1
                if activate_last_layer:
                    activation = F.relu
                else:
                    activation = lambda x: x
            else:
                out_channels = hidden_channels
                activation = F.relu
            self.node_layers.append(
                MCHCLinkSageConv(in_channels=input_channels, out_channels=out_channels, aggregation=aggregation,
                                 normalize=normalize, activation=activation, project = project))



    def forward(self, x, edge_index, class_edge_index, physical_edge_index = None):
        for layer in self.node_layers:
            x = layer(x, edge_index, class_edge_index)
        return x
