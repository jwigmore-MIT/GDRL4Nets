# Links on pytorch_geometric and torchrl
https://github.com/pytorch/rl/discussions/2243
https://github.com/facebookresearch/BenchMARL/blob/a9309159d6d46d099bd3d395ef1c80a5227b007e/benchmarl/models/gnn.py#L365


We have to handle non-batched data e.g. a single environment in a single timestep
and batched data e.g. multiple environments in a single timestep or a batch of data from a single environment

torchrl handles batches of data by stacking tensors along the batch dimension
pytorch_geometric handles batches of data by creating a super graph with disconnected edges between different graphs

So how should data be stored? Probably in a dense reprensation with stacks, and then everytime we need to pass a batch
of data through a GNN we convert to a pyg batch object
Best case scenario would be to store everything as a pyg data object


During online interaction, only the actor is needed and in this case it is currently a single environment meaning a
single observation in the form of an observation (td["observation")] and an edge_index (td["adj_sparse"]) tensor is passed

The PyG GNNs can operate on tensors when it corresponds to a single data point (graph = (observation, edge_index))

Once we do a rollout, we then have a batch of data as a tensordict. All items in the tensordicts are tensors with the same
shape[0] (batch dimension).

We then need to perform the following operations:
1. On the entire data batch, compute the advantages using GAE ((tensor -> batch -[GNN]-> tensor))
    This requires passing the entire batch -- (td["observation"], td["adj_sparse"]) and (td["next_observation"], td["next_adj_sparse"])
    through the critic. This returns the corresponding state_value and next_state_value tensors
    In this case we need to create two batch objects, one for the current state and one for the next state
    Once we have state_value and next_state_value tensors, can simply call the GAE function to get the advantages


![GNN_Performance_3NodeLineGraph.png](images%2FGNN_Performance_3NodeLineGraph.png)![img.png](img.png)
Demonstrates the performance of a GNN on a 3 node line graph. Imitation learning is used to train the GNN policy.
Where the agent fail

