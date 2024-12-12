# Implementation List
## Multiclass Multihop Class
Main idea: Simulator for a multiclass multihop 


## Backpressure Agent
Main idea: Backpressure algorithm that can function as a torchrl agent

## Basic GNN Agent
Main idea: Use a simple GCN n



## Queueing Network Simulation Environment needs:

Initialized by a set of dictionaries that specify:
   1. the number of nodes (int or list of nodes) -> N = number of nodes
   2. the arrival distribution for each class of traffic (str)
   3. the service distribution for each class of traffic (str)
   4. The adjacency for each link (list of dicts("start":int, "end":int, "rate":float)) -> M = number of links
   5. The arrival and destination for each class of traffic (list of dicts("source":int, "destination":int, "rate":float)) -> K = number of classes


The step function takes as input a TensorDict containing "action" which is an (M,K) array specifying for each (m,k),
the number of class k packets to transmit from the start of link m to the end of link m. 

Within the step function, need to call _get_valid_action() which will ensure that:
    1. The number of class k packets transmitted from Q[i,k] is less than or equal to the number of packets in Q[i,k]
        - Need to ensure sum of class k packets transmitted from links with start node i is less than or equal to the number of packets in Q[i,k]
    2. The number of packets transmitted over link m is less than the capacity
        - Need to ensure sum of class k packets transmitted over link m is less than the capacity of link m
    3. For each link and class, we don't transmit class k packets over link m if there is not a path from the end of link m to the destination of class k


---
Backpressure implementation
- Completed, see backpressure.py for algorithm and torchrl actor

---
# Problem Representatation

The network can be represented as a graph G = (V,E, X_v, X_e) and a Traffic Matrix  where:
- V is the set of nodes
- E is the set of edges
- X_v is the set of node features
- X_e is the set of edge features

(V, E)




