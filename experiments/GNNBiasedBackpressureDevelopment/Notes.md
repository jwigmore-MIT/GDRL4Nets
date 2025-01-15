# To Do
1. Implement and test Biased Backpressure
2. Implement and test Shortest Path Biased Backpressure
3. Implement and test Graph Transformation
4. Modify old GNN architecture for Biased Backpressure
5. Create update algorithm for biased backpressure
6. Implement and test Biased Backpressure with GNN


## Graph Transformation
Needs to be an edge graph. Can re-use much of the MCMHPygLinkGraphTransform code except for getting queue sizes we get
arrival rates for class k and distance to destination node k for each node.

## Biased Backpressure
Modify MultiClassMultiHopBackpressure to include a bias term for each edge and class.
Can I store this as a sparse N x N x K matrix?



