from tensordict.nn import TensorDictModule
from torch_geometric_development.conversions import lazy_stacked_tensor_dict_to_batch as lstd2b, tensor_dict_to_data as td2d
import torch
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.pool import global_add_pool
from torchrl.modules import ProbabilisticActor, ValueOperator, ActorCriticWrapper, MaskedCategorical
from torchrl.envs import ExplorationType
from torchrl.data import CompositeSpec
from torchrl_development.actors import MaskedOneHotCategoricalTemp
from torchrl.modules import ReparamGradientStrategy


class GNN_Critic(torch.nn.Module):

    def __init__(self, in_channels = 1, out_channels = 1, hidden_channels = 16):
        super(GNN_Critic, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        # Need aggregation layer
        self.aggregation = global_add_pool

    def forward(self, x, edge_index, batch, batch_size):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        #x = torch.relu(x)
        x = self.aggregation(x, batch, batch_size)
        return x

class GNNTensorDictModule(TensorDictModule):

    def __init__(self, model, in_keys = ["x", "edge_index"], out_keys = ['logits'], forward_batch = False, keep_td = False):
        super(GNNTensorDictModule, self).__init__(model,
                                                  in_keys=in_keys,
                                                  out_keys=out_keys)
        self.model = model
        self.forward_batch = forward_batch
        if keep_td:
            self.forward = self.forward_keep
        else:
            self.forward = self.forward_convert

    def forward_keep(self, td):
        # TODO: Implement forward pass that keeps the tensordicts as is, but need to figure out how to combine the graphs for batch processing
        pass

    def forward_convert(self, td):
        "input is now a tensordict that will contain a graph"
        if td.get("batch", None) is not None:
            pass
        else:
            if td.dim() == 1:
                batch = lstd2b(td)
                # x = batch['x']
                # edge_index = batch['edge_index']
                # num_graphs = batch.num_graphs
                # batch_batch = batch.batch
                # batch_size = batch.batch_size
            else:
                batch = td2d(td)
                batch.num_graphs = 1
                # x = td["x"]
                # edge_index = td["edge_index"]
                # num_graphs = 1

        #batch = lstd2b(td)
        x = batch["x"]
        edge_index = batch["edge_index"]
        if self.forward_batch:
            z = self.module(x, edge_index, batch.batch, batch.batch_size)
        else:
            z = self.module(x, edge_index)
        # Z should have the shape [N*B, F] where N is the number of nodes, B is the batch size and F is the number of output features
        # We need to reshape Z to [B, N, F]
        z = z.view(batch.num_graphs, -1, z.size(-1))
        # append 0s to make z have shape (B, N+1, F)
        if not self.forward_batch:
            z = torch.cat([-1*torch.ones(z.size(0), 1, z.size(-1)),z], dim=1)
        td[self.out_keys[0]] = z.squeeze(-1)
        return td

def create_GNN_Actor_Critic(actor_network, critic_network, in_keys = ["x", "edge_index"], action_spec = None, temperature = 1.0):

    actor_module = GNNTensorDictModule(actor_network, in_keys=in_keys, out_keys=['logits'])  # create actor
    actor_module = ProbabilisticActor(actor_module,
                                      distribution_class=MaskedOneHotCategoricalTemp,
                                      in_keys=["logits", "mask"],
                                      distribution_kwargs={"grad_method": ReparamGradientStrategy.RelaxedOneHot,
                                                           "temperature": temperature},  # {"temperature": 0.5}
                                      spec = CompositeSpec(action=action_spec),
                                      return_log_prob=True,
                                      default_interaction_type = ExplorationType.RANDOM)


    value_module = GNNTensorDictModule(critic_network, in_keys=in_keys, out_keys = ['state_value'], forward_batch = True)

    return ActorCriticWrapper(actor_module, value_module)





