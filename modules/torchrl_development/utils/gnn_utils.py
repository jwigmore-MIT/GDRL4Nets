import torch
from torch_geometric.data import Batch

def tensors_to_batch(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        class_edge_index: torch.Tensor = None,
        physical_edge_index: torch.Tensor = None,
        K = None,
        ):
    """
    Coverts x [B, N, F], edge_index [2, MK], and class_edge_index [2, N(K(K-1)] to a batch object
    :param x:
    :param edge_index:
    :return:
    """
    if K is None:
        raise ValueError("K must be provided")
    batch_size = x.shape[0]
    nodes_per_graph = x.shape[1]
    b = torch.arange(batch_size, device=x.device)
    graphs = Batch()
    graphs.K = K
    graphs.x = x.view(-1, x.shape[-1]) # essentially stacks the nodes to create a [B*N, F] tensor
    graphs.batch = torch.repeat_interleave(b, nodes_per_graph)

    n_edges = edge_index.shape[-1]
    batch = torch.repeat_interleave(b, n_edges)
    edge_index = edge_index.transpose(0,1).contiguous().view(2,-1)
    batch_edge_index = edge_index + batch*nodes_per_graph
    graphs.edge_index = batch_edge_index
    if class_edge_index is not None:
        n_class_edges = class_edge_index.shape[-1]
        batch = torch.repeat_interleave(b, n_class_edges)
        class_edge_index = class_edge_index.transpose(0,1).contiguous().view(2,-1)
        batch_class_edge_index = class_edge_index + batch*nodes_per_graph
        graphs.class_edge_index = batch_class_edge_index
    if physical_edge_index is not None:
        n_physical_edges = physical_edge_index.shape[-1]
        batch = torch.repeat_interleave(b, n_physical_edges)
        physical_edge_index = physical_edge_index.transpose(0,1).contiguous().view(2,-1)
        batch_physical_edge_index = physical_edge_index + batch*nodes_per_graph
        graphs.physical_edge_index = batch_physical_edge_index
    return graphs.to(x.device)