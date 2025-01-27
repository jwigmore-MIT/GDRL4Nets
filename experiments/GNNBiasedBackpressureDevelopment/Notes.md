# To Do
1. ~~Implement and test Biased Backpressure~~
2. ~~Implement and test Shortest Path Biased Backpressure~~
3. ~~Implement and test Graph Transformation~~
4. ~~Create Attention based GNN architecture~~
5. Test Attention Based GNN architecture... How?
5. Create update algorithm for biased backpressure
6. Implement and test Biased Backpressure with GNN

Extra:
1. Modify Environment to have multiple source nodes and with individual arrival rates for each class - all share the 
   same destination node
2. 



## Graph Transformation
~~Needs to be an edge graph. Can re-use much of the MCMHPygLinkGraphTransform code except for getting queue sizes we get
arrival rates for class k and distance to destination node k for each node.~~

We actually don't need to convert to an edge graph. We need:
1. Node-Features: (N,K,F) where F is the number of features
2. Edge Features: (M,F) where M is the number of edges and F is the number of features
3. Edge Index: (M,2) where M is the number of edges and each row is the source and destination node index


Node Features:
1. X[n,k,0] = arrival rate of class k at node n
2. X[n,k,1] = distance to destination node k from node n

Edge Features:
1. X[m,0] = average link rate for edge m

What if we wanted to include edge features for each class, e.g.
1. X[m,k,0] = average link rate for edge m for class k (would be the same for all classes in the current model)
2. X[m,k,1] = bias for edge m for class k

I think we should be able to get this simply by calling env.get_observation() and then reshaping the output

IMPLEMENTATION: We do this all internally in the env.get_rep() function
returns a tensordict with keys:
   X: N,K,Fn tensor
   edge_index: M,2 tensor
   edge_attr: M,Fe tensor

## Attention Based GNN
First implemented a scaled dot product attention layer
`SPDA_layer(nn.Module)`

Implemented the convolution and can build deep models with DeeperGCN framework

## Testing NodeAttentionGNN
If I want to use supervised learning, what is a per link/class value that I can predict?
1. Average utilization of each link for each class under backpressure? I.e. predict the action frequency
2. 


## Biased Backpressure
Modify MultiClassMultiHopBackpressure to include a bias term for each edge and class.
Can I store this as a sparse N x N x K matrix?

## Class Information:
Should we group classes purely based on destination? 
For example, class k arrivals can arrive to multiple different nodes at different rates for each node

The biggest benefit would be state-space compression when two arrival classes share the same destination

This is important for the GNN architecture because it requires splitting into a class graph

Class k used to be: (arrival_node, destination_node, rate)


## Reward Attribution 
Right now every link gets the same cost.
We could try to attribute individual costs to each link-class based on the
average queue size at each end node of the link for that class