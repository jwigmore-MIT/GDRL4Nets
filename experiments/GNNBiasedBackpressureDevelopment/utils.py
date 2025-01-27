import torch
from torch_geometric.data import Batch
import networkx as nx
import matplotlib.pyplot as plt
import math
import torch_geometric as pyg
import matplotlib as mpl
def tensors_to_batch(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        K = None,
        ):
    """
    Coverts x [B, N, F], edge_index [2, MK], and class_edge_index [2, N(K(K-1)] to a batch object

    Currently only works if all graphs have the same number of nodes and edges
    :param x:
    :param edge_index:
    :return:
    """

    if K is None:
        raise ValueError("K must be provided")
    batch_size = x.shape[0] # number of graphs/different network "states"
    nodes_per_graph = x.shape[1] # only works if all graphs have the same number of nodes
    edges_per_graph = edge_index.shape[-1] # only works if all graphs have the same number of edges

    b = torch.arange(batch_size, device=x.device)
    graphs = Batch()
    graphs.K = K

    graphs.x = x.view(-1, x.shape[-1]) # essentially stacks the nodes to create a [B*N, F] tensor
    graphs.batch = torch.repeat_interleave(b, nodes_per_graph)

    batch = torch.repeat_interleave(b, edges_per_graph)
    edge_index = edge_index.transpose(0,1).contiguous().view(2,-1)
    batch_edge_index = edge_index + batch*nodes_per_graph
    graphs.edge_index = batch_edge_index


    return graphs.to(x.device)

def plot_nx_graph(graph, edge_attr = None, node_attr = None,  K = None, title = "", subtitle = "", erange = None, vrange = None):
    """
    Plots k networkx graphs with edge attributes for each k

    :param graph:
    :param edge_attr:
    :param epoch:
    :param reward:
    :return:
    """
    if edge_attr.dim() > 2:
        edge_attr = edge_attr.squeeze(-1)
    if K is None:
        K = edge_attr.shape[-1]

    rows = int(math.ceil(math.sqrt(K)))
    cols = int(math.ceil(K / rows))
    # create fig and ax with k subplots, as a grid
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    cmp = plt.cm.viridis
    if erange is not None:
        emin_val, emax_val = erange
    else:
        emin_val, emax_val = -10,10
    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin, vmax = 0,10
    enorm = mpl.colors.Normalize(vmin=emin_val, vmax=emax_val)
    vnorm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    all_axes = axes.flatten() if K > 1 else [axes]
    for k, ax in enumerate(all_axes):
        # row = k // cols
        # col = k % cols
        # ax = axes[row, col] if K > 1 else axes
        # Create new graph with only the kth edge attribute
        # k_edge_attribute = {}
        # for m in range(len(edge_attr)):
        #     k_edge_attribute[tuple(graph.edge_index[:,m].tolist())] = round(edge_attr[m,k].item(),2)
        graph.edge_attr = edge_attr[:,k].unsqueeze(-1)
        graph.node_attr = node_attr[:,k].unsqueeze(-1)
        nx_graph = pyg.utils.to_networkx(graph, edge_attrs = ["edge_attr"], node_attrs = ["node_attr"], to_undirected=False)
        # for m,edge in enumerate(nx_graph.edges()):
        #     edge[2]["edge_attr"] = round(edge_attr[m,k].item(),2)
        pos = nx.spring_layout(nx_graph, iterations=100, seed = 0)

        # plot the graph with edge labels
        if node_attr is not None:
            node_color = [node[1]["node_attr"][0] for node in nx_graph.nodes(data=True)]
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=300, alpha = 0.5, node_color = node_color, cmap = cmp, vmin = vmin, vmax = vmax)
            nx.draw_networkx_labels(nx_graph, pos, ax=ax, labels={i: f"v{i}:{node_attr[i, k].item():.{1}f}" for i in range(node_attr.shape[0])})

            # nx.draw_networkx_labels(nx_graph, pos, ax = ax, labels = labels)
        else:
            nx.draw_networkx_nodes(nx_graph, pos, ax=ax, node_size=100)
            nx.draw_networkx_labels(nx_graph, pos, ax = ax)
        if edge_attr is not None:
            edge_color = [edge[2]["edge_attr"][0] for edge in nx_graph.edges(data=True)]
            # create edge color and normalize it by the min and max values
            # edge_color = [(edge[2]["loc"]-min_val)/(max_val - min_val) for edge in nx_graph.edges(data=True)]
            # edge_color = norm(edge_color)
            edges = nx.draw_networkx_edges(nx_graph, pos, ax = ax, connectionstyle="arc3,rad=0.2",
                                           edge_color = edge_color, edge_cmap = cmp, edge_vmin = emin_val, edge_vmax = emax_val)
            labels = {edge: round(attr[0],2) for edge, attr in nx.get_edge_attributes(nx_graph, "edge_attr").items()}
            nx.draw_networkx_edge_labels(nx_graph, pos, edge_labels=labels, ax = ax, connectionstyle="arc3,rad=0.2")
        else:
            nx.draw_networkx_edges(nx_graph, pos, ax = ax, connectionstyle="arc3,rad=0.2")
        ax.set_title(f"Class {k}")
    pc = mpl.collections.PatchCollection(edges, cmap=cmp, norm=enorm)
    pc.set_array(edge_color)
    fig.suptitle(f"{title}\n{subtitle}")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.colorbar(pc, ax=axes, orientation='vertical', label = "Edge Weight", )

    # plt.show()
    return fig, axes