import torch
from torch_geometric.data import Data

edge_index_tuple = torch.tensor([[0,1], # edge 0 connects node 0 to node 1
                                 [1,0],      # edge 1 connects node 1 to node 0
                                 [1,2],      # edge 2 connects node 1 to node 2
                                 [2,1]],    # edge 3 connects node 2 to node 1
                                dtype=torch.long)

edge_index = edge_index_tuple.t().contiguous() # Transpose edge index

x = torch.tensor([[2], [1], [3]], dtype=torch.float) # Node features

data = Data(x=x, edge_index=edge_index)

print(data.keys()) # tells us what attributes are stored in the data object

print(data['x']) # Node features

print(data['edge_index']) # Edge index

print(data.num_nodes) # Number of nodes

print(data.num_edges) # Number of edges

