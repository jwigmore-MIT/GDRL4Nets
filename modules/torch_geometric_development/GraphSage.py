from torch_geometric.nn.models import GraphSAGE
from tensordict.nn import TensorDictModule
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b, tensor_dict_to_data as td2d
import torch

class GraphSageModule(TensorDictModule):

    def __init__(self, in_keys, out_keys, in_channels = 1, hidden_channels = 2 , num_layers = 2, aggr='max'):
        self.model = GraphSAGE(in_channels, hidden_channels, num_layers, aggr)

        super(GraphSageModule, self).__init__(self.model, in_keys, out_keys)

    def forward(self, td):
        if td.batch_size.__len__() == 1:
            data = lstd2b(td[self.in_keys[0]])
        else:
            data = td2d(td[self.in_keys[0]])
            data.num_graphs = 1
        x = data["x"]
        edge_index = data["edge_index"]
        z = self.model(x, edge_index)
        # Z should have the shape [N*B, F] where N is the number of nodes, B is the batch size and F is the number of output features
        # We need to reshape Z to [B, N, F]
        z = z.view(data.num_graphs, -1, z.size(-1))
        # append 0s to make z have shape (B, N+1, F)
        z = torch.cat([-1*torch.ones(z.size(0), 1, z.size(-1)),z], dim=1)
        td["logits"] = z.squeeze(-1)
        return td
