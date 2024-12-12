
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torchrl.envs import check_env_specs
import torch
from torch_geometric.data import Batch


def tensors_to_batch(
        x: torch.Tensor,
        edge_index: torch.Tensor):
    """
    Coverts x [B, N, F] and edge_index [2, E] to a batch object
    :param x:
    :param edge_index:
    :return:
    """
    batch_size = x.shape[0]
    nodes_per_graph = x.shape[1]
    b = torch.arange(batch_size, device=x.device)
    graphs = Batch()
    graphs.x = x.view(-1, x.shape[-1]) # essentially stacks the nodes to create a [B*N, F] tensor
    graphs.batch = torch.repeat_interleave(b, nodes_per_graph)
    n_edges = edge_index.shape[-1]
    batch = torch.repeat_interleave(b, n_edges)
    edge_index = edge_index.transpose(0,1).contiguous().view(2,-1)
    batch_edge_index = edge_index + batch
    graphs.edge_index = batch_edge_index
    return graphs.to(x.device)



file_path = "../env2.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)


net = MultiClassMultiHop(**env_info)

check_env_specs(net)




