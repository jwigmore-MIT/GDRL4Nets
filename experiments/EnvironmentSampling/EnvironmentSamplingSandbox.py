import networkx as nx
from graph_env_creators import make_line_graph, make_ring_graph, create_grid_graph
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.transforms import ToDense
from get_graph_characteristics import get_graph_characteristics
import torch
import matplotlib.pyplot as plt
import numpy as np

# create a graph
adj, arrival_dist, arrival_rate, service_dist, service_rates = create_grid_graph(rows = 4, columns= 4,  arrival_rate = 0.5, service_rate = 1.0)

# get the sparse representation of the adjacency matrix
adj_sparse = dense_to_sparse(torch.Tensor(adj))[0]

# convert the sparse representation to a dense representation
adj2 = to_dense_adj(adj_sparse).squeeze()

# get the transpose of the adjacency matrix - more easily understood
adj_sparse_T = adj_sparse.T

# create networkx graph from the adjacency matrix
G = nx.convert_matrix.from_numpy_array(adj)

# plot G
plt.figure()
nx.draw(G, with_labels=True)
plt.show()

# Get graph characteristics
graph_chars = get_graph_characteristics(G)

