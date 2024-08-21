import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tensordict import TensorDict, LazyStackedTensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
from copy import deepcopy
from torch_geometric.loader import DataLoader
from tensordict.prototype import tensorclass
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b, tensor_dict_to_data as td2d



class MaxWeightUpdate(torch.nn.Module):
    def __init__(self, init_weights = torch.Tensor([1,-1])):
        super(MaxWeightUpdate, self).__init__()
        self.weights = torch.nn.Parameter(init_weights).unsqueeze(0)

    def forward(self, z):
        return z @ self.weights.transpose(0,1)
        #return torch.tanh(z @ self.weights.transpose(0,1))



class MaxWeightConv(MessagePassing):

    def __init__(self, init_weights: torch.Tensor = torch.Tensor([1,-1])):
        super(MaxWeightConv, self).__init__(aggr='max', flow='source_to_target')
        '''
        aggr is max because we are taking the max of the messages from the neighbors
        flow is source_to_target because we are sending messages from the source nodes to the target nodes
        '''
        # self.update_module =  MaxWeightUpdate(init_weights)

    def forward(self, x, edge_index):
        # x has shape [B,N, F], where B is the size of the batch, N is the number of nodes and F the number of input features
        # edge_index has shape [B, 2, E] where E is the number of edges

        # step 1: add self loops to the adjacency matrix (if needed)
        edge_index, _ = add_self_loops(edge_index)

        # step 2: propagate the messages which calls message, aggregate, and update functions
        agg_out = self.propagate(edge_index, x=x) #

        # Step 3 Call self.update_module to update the node features
        # z = self.update_module(torch.concat([x,agg_out], dim=1))

        # step 3: update the node features
        return agg_out

class MaxWeightGNN(torch.nn.Module):
    def __init__(self, init_weights = torch.Tensor([[1,-1]])):
        super(MaxWeightGNN, self).__init__()
        if init_weights.dim() == 1:
            init_weights = init_weights.unsqueeze(0)
        self.layer = MaxWeightConv()
        self.weights = torch.nn.Parameter(init_weights)


    def forward(self, x, edge_index):
        # x should have shape [N,2] where N is the number of nodes and the first column is the Q values and the second column is the Y values
        # for MaxWeight we need to multiple the features of the nodes together
        x = x.prod(dim=1, keepdim=True)
        agg_out = self.layer(x, edge_index)
        #z = torch.mul(torch.concat([x, agg_out], dim=1),self.weights)
        z = torch.cat([x, agg_out], dim=1) @ self.weights.transpose(0,1)
        return z


class GNNTensorDictModule(TensorDictModule):

    def __init__(self, model, in_keys = ["graph"], out_keys = ['logits'], keep_td = False):
        super(GNNTensorDictModule, self).__init__(model,
                                                  in_keys=in_keys,
                                                  out_keys=out_keys)
        self.model = model
        if keep_td:
            self.forward = self.forward_keep
        else:
            self.forward = self.forward_convert

    def forward_keep(self, td):
        # TODO: Implement forward pass that keeps the tensordicts as is, but need to figure out how to combine the graphs for batch processing
        pass

    def forward_convert(self, td):
        "input is now a tensordict that will contain a graph"
        if td.batch_size.__len__() == 1:
            batch = lstd2b(td[self.in_keys[0]])
        else:
            batch = td2d(td[self.in_keys[0]])
            batch.num_graphs = 1
        x = batch["x"]
        edge_index = batch["edge_index"]
        z = self.module(x, edge_index)
        # Z should have the shape [N*B, F] where N is the number of nodes, B is the batch size and F is the number of output features
        # We need to reshape Z to [B, N, F]
        z = z.view(batch.num_graphs, -1, z.size(-1))
        # append 0s to make z have shape (B, N+1, F)
        z = torch.cat([-1*torch.ones(z.size(0), 1, z.size(-1)),z], dim=1)
        td["logits"] = z.squeeze(-1)
        return td




class GNNTensorDictModule1(TensorDictModule):

    def __init__(self, model, in_keys = ["x", "edge_index"], out_keys = ['logits'], keep_td = False):
        super(GNNTensorDictModule1, self).__init__(model,
                                                  in_keys=in_keys,
                                                  out_keys=out_keys)
        self.model = model
        if keep_td:
            self.forward = self.forward_keep
        else:
            self.forward = self.forward_convert

    def forward_keep(self, td):
        # TODO: Implement forward pass that keeps the tensordicts as is, but need to figure out how to combine the graphs for batch processing
        pass

    def forward_convert(self, td):
        "input is now a tensordict that will contain a graph"
        if td.dim() == 1:
            batch = lstd2b(td)
        else:
            batch = td2d(td)
            batch.num_graphs = 1
        #batch = lstd2b(td)
        x = batch["x"]
        edge_index = batch["edge_index"]
        z = self.module(x, edge_index)
        # Z should have the shape [N*B, F] where N is the number of nodes, B is the batch size and F is the number of output features
        # We need to reshape Z to [B, N, F]
        z = z.view(batch.num_graphs, -1, z.size(-1))
        # append 0s to make z have shape (B, N+1, F)
        z = torch.cat([-1*torch.ones(z.size(0), 1, z.size(-1)),z], dim=1)
        td["logits"] = z.squeeze(-1)
        return td
