from torchrl.envs import ExplorationType
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
import torch
import torch.nn as nn
import torch.distributions as D
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,
)
from torch_geometric.data import Batch

from typing import Tuple
from torchrl.data.tensor_specs import CompositeSpec
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch

class GNN_TensorDictModule(TensorDictModule):

    def __init__(self, module, x_key="X", edge_index_key="edge_index", class_edge_index_key = "class_edge_index", out_key= "probs"):
        super(GNN_TensorDictModule, self).__init__(module=module, in_keys=[x_key, edge_index_key, class_edge_index_key], out_keys=out_key)
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.class_edge_index_key = class_edge_index_key
        self.out_key = out_key

    def forward(self, input):
        # Three different cases to handle: 1) input is a tensordict of size 1, 2) input is a tensordict of size > 1, 3) input is a batch
        if isinstance(input, TensorDict):
            if input[self.x_key].dim() < 3:  # Case 1: input is a tensordict of size 1
                input[self.out_key] = self.module(input[self.x_key], input[self.edge_index_key], input[self.class_edge_index_key]).squeeze(-1)
                return input
            else: # case 2: input is a tensordict of size > 1
                batch_graph = tensors_to_batch(input[self.x_key], input[self.edge_index_key], input[self.class_edge_index_key])
                input[self.out_key] = self.module(batch_graph.x, batch_graph.edge_index, batch_graph.class_edge_index).view(batch_graph.batch_size,-1)
                return input
        elif isinstance(input, Batch): # case 3
            return self.module(input.x, input.edge_index, input.batch)


class Policy_Module(nn.Module):
    def __init__(self, gnn_module):
        super().__init__()
        self.gnn_module = gnn_module

    def forward(self, x_in, edge_index, edge_class_index, M):
        logits = self.gnn_module(x_in, edge_index)
        if self.training:
            x = self.softmax(logits, edge_index)
        else:
            x = self.argmax(logits, edge_index)
        # check if x contains nan
        if torch.isnan(x).any():
            print("x contains nan")
        return x, logits
class GNN_ActorTensorDictModule(GNN_TensorDictModule):

    def __init__(self, module = None, x_key = "X", edge_index_key = "edge_index", class_edge_index_key = "class_edge_index",
                 physical_edge_index_key = "physical_edge_index", out_keys = ["probs", "logits"]):
        # module will be the MCHCGraphSage

        super(GNN_ActorTensorDictModule, self).__init__(
            module=module, x_key=x_key,
            edge_index_key=edge_index_key, class_edge_index_key = class_edge_index_key,
            out_key=out_keys)
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.class_edge_index_key = class_edge_index_key
        self.physical_edge_index_key = physical_edge_index_key
        self.outs_key = out_keys
        self.small_logits = torch.log(torch.Tensor([0.001]))

    def forward(self, input):
        if isinstance(input, TensorDict):
            K = input["Q"].shape[-1]
            if input[self.x_key].dim() < 3: # < 3 # batch size is 1, meaning $\tilde X$ has shape [NK,F] an
                """
                Batch size is 1, meaning $\tilde X$.shape = (N*K,F) and module(inputs).shape = (M*K,1)
                We need to reshape the output of the module to have shape (N,K)
                """
                logits = self.module(input[self.x_key],
                                     input[self.edge_index_key],
                                     input[self.class_edge_index_key],
                                     input[self.physical_edge_index_key]
                                     )
                # now concat a tensor of shape logits.shape[0] with value  = self.small_logits to the first dimension of logits
                logits = logits.view(-1,K)
                logits = torch.cat(( self.small_logits.repeat(logits.shape[0],1),logits), dim = 1)
                probs = torch.softmax(logits, dim = -1)
                input[self.outs_key[1]] = logits.squeeze(-1)
                input[self.outs_key[0]] = probs.squeeze(-1)
            else:
                batch_graph = tensors_to_batch(input[self.x_key], input[self.edge_index_key],
                                               input[self.class_edge_index_key], input[self.physical_edge_index_key], K=K)
                logits = self.module(batch_graph.x, batch_graph.edge_index,
                                     batch_graph.class_edge_index, batch_graph.physical_edge_index).view(batch_graph.batch_size, -1, K)
                logits = torch.cat((self.small_logits.expand(logits.shape[0],logits.shape[1],1), logits), dim = -1)
                probs = torch.softmax(logits, dim = -1)
                input[self.outs_key[0]] = probs.view(batch_graph.batch_size, -1, K+1)
                input[self.outs_key[1]] = logits.view(batch_graph.batch_size, -1, K+1)
            return input
        elif isinstance(input, Batch):
            logits = self.module(input.x, input.edge_index, input.class_edge_index, input.physical_edge_index).view(input.batch_size,-1, input.K)
            logits = torch.cat((self.small_logits.expand(logits.shape[0],logits.shape[1],1), logits), dim = -1)
            probs = torch.softmax(logits, dim = -1)
            return logits, probs

