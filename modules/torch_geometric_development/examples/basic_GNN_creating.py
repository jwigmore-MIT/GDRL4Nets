import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from tensordict import TensorDict, LazyStackedTensorDict
from tensordict.nn import TensorDictModule
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.data import Data, Batch
from copy import deepcopy
from torch_geometric.loader import DataLoader
from tensordict.prototype import tensorclass


"""
The message passing base class takes care of message passing propagation.  The user only needs to define 
1. the message function i.e. \phi
2. the aggregation function i.e. \bigoplus
3. the update function i.e. \sigma

For the single-layer MaxWeight GNN model, the message function is the identity function,
the aggregation function is the max function, and the update function is:
\sigma(z,z_{ag} = \text{tanh}(w_1z + w_2z_{ag})

"""



# First we implement the update function
class MaxWeightUpdate(torch.nn.Module):
    def __init__(self):

        super(MaxWeightUpdate, self).__init__()
        self.weights = torch.nn.Parameter(torch.Tensor([1,-1])).unsqueeze(0)

    def forward(self, z):

        return torch.tanh(z @ self.weights.transpose(0,1))


def MaxWeightUpdateNN_test():
    update = MaxWeightUpdate()
    z = torch.Tensor([4,1])
    print("Testing the MaxWeightUpdateNN class")
    print(f"Input: {z}")
    print(f"Output: {update(z)}")



# Now we implement the message passing class

class MaxWeightConv(MessagePassing):

    def __init__(self):
        super(MaxWeightConv, self).__init__(aggr='max', flow='source_to_target')
        '''
        aggr is max because we are taking the max of the messages from the neighbors
        flow is source_to_target because we are sending messages from the source nodes to the target nodes
        '''
        self.update_module =  MaxWeightUpdate()

    def forward(self, x, edge_index):
        # x has shape [B,N, F], where B is the size of the batch, N is the number of nodes and F the number of input features
        # edge_index has shape [B, 2, E] where E is the number of edges

        # step 1: add self loops to the adjacency matrix (if needed)
        edge_index, _ = add_self_loops(edge_index)

        # step 2: propagate the messages which calls message, aggregate, and update functions
        agg_out = self.propagate(edge_index, x=x) #

        # Step 3 Call self.update_module to update the node features
        z = self.update_module(torch.concat([x,agg_out], dim=1))

        # step 3: update the node features
        return z

class MaxWeightGNN(torch.nn.Module):
    def __init__(self):
        super(MaxWeightGNN, self).__init__()

        self.layer = MaxWeightConv()

    def forward(self, x, edge_index):

        z = self.layer(x, edge_index)
        if self.train:
            return z.softmax(-1)
        else:
            return z.argmax(-1)


class MaxWeightGNNTensorDictModule(TensorDictModule):

    def __init__(self, in_keys = ["graph"], out_keys = ['action']):
        module = MaxWeightGNN()
        super(MaxWeightGNNTensorDictModule, self).__init__(module,
                                                           in_keys=in_keys,
                                                           out_keys=out_keys)


    def forward(self, td):
        "input is now a tensordict that will contain"
        x = td[self.in_keys[0]]["x"]
        edge_index = td[self.in_keys[0]]["edge_index"]
        z = self.module(x, edge_index)
        td["logits"] = z
        return td




def TestMaxWeightConv():
    conv = MaxWeightConv()
    q = torch.Tensor([[2],[10]])
    y = torch.Tensor([[3],[1]])
    x = torch.multiply(q,y)

    edge_index = torch.tensor([[0,1],# edge 0 connects node 0 to node 1
                               [1,0],      # edge 1 connects node 1 to node 0
                               ], dtype=torch.long).t().contiguous()

    print("Testing the MaxWeightConv class")
    print(f"Input node features: {x}")
    print(f"Input edge index: {edge_index}")
    print(f"Output node features: {conv(x, edge_index)}")


def TestMaxWeightGNN():
    gnn = MaxWeightGNN()
    q = torch.Tensor([[2],[10]])
    y = torch.Tensor([[3],[1]])
    x = torch.multiply(q,y)

    edge_index = torch.tensor([[0,1],# edge 0 connects node 0 to node 1
                               [1,0],      # edge 1 connects node 1 to node 0
                               ], dtype=torch.long).t().contiguous()

    print("Testing the MaxWeightGNN class")
    print(f"Input node features: {x}")
    print(f"Input edge index: {edge_index}")
    print(f"Output node features: {gnn(x, edge_index)}")

"""
Need a way to convert between TensorDicts and torch_geometric.data.Data objects
"""

def tensor_dict_to_data(td):
    """Convert a TensorDict object to a torch_geometric.data.Data object
    Can only handle a TensorDict with a single set of graph attributes at a time
    """
    return Data(**{key: td[key].squeeze(0) for key in td.keys()}) # squeeze the batch dimension

def data_to_tensor_dict(data):
    """
    Convert a torch_geometric.data.Data object to a TensorDict object
    Can only handle a single graph at a time
    :param data:
    :return:
    """
    return TensorDict({key: data[key].unsqueeze(0) for key in data.keys()}, batch_size=1) # add the batch dimension


def stacked_tensor_dict_to_data(td):
    """
    Converts a LazyStackedTensorDict object to a torch_geometric.loader.DataLoader object
    :param td:
    :return:
    """
    # Check if dim is 1
    if td.dim() > 1:
        raise ValueError("The input tensor dict has dimension greater than 1.  Cannot convert to DataLoader object.")
    # First check if the LazyStackedTensorDict object is a batch of graphs or a single graph
    if td.batch_size[0] == 1:
        return tensor_dict_to_data(td)
    else:
        return DataLoader([tensor_dict_to_data(td[i]) for i in range(td.batch_size[0])], batch_size=td.batch_size[0])



def tensor_dict_to_data(td):
    return Data(**{key: td[key] for key in td.keys()})

def data_to_tensor_dict(data):
    return TensorDict({key: data[key] for key in data.keys()}, [])

def lazy_stacked_tensor_dict_to_batch(lstd):
    return Batch.from_data_list([Data(**{key: lstd[key][i] for key in lstd.keys()}) for i in range(lstd.batch_size)])

def batch_to_lazy_stacked_tensor_dict(batch):
    return LazyStackedTensorDict({key: torch.stack([batch[i][key] for i in range(batch.num_graphs)]) for key in batch[0].keys()}, batch.num_graphs)



def TestMaxWeightGNNTensorDictModule():
    """
    For the A GNNTensorDictModule, the input is a TensorDict object that contains a key "graph"
    that contains another tensor dict object with  keys "x" and "edge_index"
    This "graph" tensor dict is analogous to the "data" object in the torch_geometric.data.Data object
    This way, we should be able to batch multiple graphs together
    :return:
    """


    gnn_module = MaxWeightGNNTensorDictModule(in_keys=["graph"], out_keys=['action'])


    q1 = torch.Tensor([[2], [10]])
    y1 = torch.Tensor([[3], [1]])
    x1 = torch.multiply(q1, y1).unsqueeze(0)


    edge_index1 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                               [1, 0],  # edge 1 connects node 1 to node 0
                               ], dtype=torch.long).t().contiguous().unsqueeze(0)
    graph_td = TensorDict({'x': x1, 'edge_index': edge_index1}, batch_size=1)
    td1 = TensorDict({'graph': graph_td}, batch_size=1)

    q2 = torch.Tensor([[1], [5]])
    y2 = torch.Tensor([[2], [1]])
    x2 = torch.multiply(q2, y2).unsqueeze(0)
    edge_index2 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                               [1, 0],  # edge 1 connects node 1 to node 0
                               ], dtype=torch.long).t().contiguous().unsqueeze(0)

    graph_td2 = TensorDict({'x': x2, 'edge_index': edge_index2}, batch_size=1)
    td2 = TensorDict({'graph': graph_td2}, batch_size =1)

    #merge the two tensor dicts
    stacked = torch.stack([td1, td2], 0).contiguous()




# @tensorclass
# class GData(Data):
#
#     def __init__(self,
#         x = None,
#         edge_index = None,
#         edge_attr= None,
#         y = None,
#         pos = None,
#         time = None,
#         **kwargs,):
#         super(GData, self).__init__(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos, time=time, **kwargs)


def TestSingleGraphConversion():
    q1 = torch.Tensor([2,10])
    y1 = torch.Tensor([1,3])
    x1 = torch.multiply(q1, y1)

    edge_index1 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                [1, 0]], # edge 1 connects node 1 to node 0
                                dtype=torch.long).t().contiguous()



    td_graph = TensorDict({'x': x1.unsqueeze(0), 'edge_index': edge_index1.unsqueeze(0)}, batch_size=1)
    graph = Data(x=x1, edge_index=edge_index1)


    td_graph_from_graph = data_to_tensor_dict(graph)
    graph_from_td_graph = tensor_dict_to_data(td_graph)
    print("Successfully converted between TensorDict and Data objects")


def NonHomogenousStackingDemo():
    """
    Want to show how to stack two TensorDict objects with keys that have different dimensions
    :return:
    """

    x1 = torch.Tensor([[2, 10], [1, 5]])
    x2 = torch.Tensor([[1, 5, 3], [5,3,1]])
    edge_index1 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                [1, 0],  # edge 1 connects node 1 to node 0
                                ], dtype=torch.long).t().contiguous()
    edge_index2 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                [1, 0],  # edge 1 connects node 1 to node 0
                                [1, 2],  # edge 2 connects node 1 to node 2
                                [2, 1],  # edge 3 connects node 2 to node 1
                                ], dtype=torch.long).t().contiguous()

    try:
        td1 = TensorDict({'x': x1, 'edge_index': edge_index1}, batch_size=[])
        td2 = TensorDict({'x': x2, 'edge_index': edge_index2}, batch_size=[])
        stacked1 = torch.stack([td1, td2], 0)
        print("Stacking with non-homogenous dimensions successful using no batch size")
    except Exception:
        print("Stacking with non-homogenous dimensions failed using no batch size")
    try:
        td1b = TensorDict({'x': x1.unsqueeze(0), 'edge_index': edge_index1.unsqueeze(0)}, batch_size=1)
        td2b = TensorDict({'x': x2.unsqueeze(0), 'edge_index': edge_index2.unsqueeze(0)}, batch_size=1)
        stacked2 = torch.stack([td1b, td2b], 0)
        print("Stacking with non-homogenous dimensions successful using batch size")
    except Exception:
        print("Stacking with non-homogenous dimensions failed using batch size")
    return stacked1, stacked2

def BatchedGraphFromTDsTest():

    # First define the node features and edge index for two graphs
    x1 = torch.Tensor([2, 10]) #features for two node graph
    x2 = torch.Tensor([[1, 5], [3, 4], [2,7]]) #features for three node graph

    edge_index1 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                [1, 0],  # edge 1 connects node 1 to node 0
                                ], dtype=torch.long).t().contiguous()

    edge_index2 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                [1, 0],  # edge 1 connects node 1 to node 0
                                [1, 2],  # edge 2 connects node 1 to node 2
                                [2, 1],  # edge 3 connects node 2 to node 1
                                ], dtype=torch.long).t().contiguous()

    # Create the TensorDict objects for the two graphs
    td_graph1 = TensorDict({'x': x1, 'edge_index': edge_index1}, [])
    td_graph2 = TensorDict({'x': x2, 'edge_index': edge_index2}, [])

    # Create Data objects for the two graphs
    graph1 = Data(x=x1, edge_index=edge_index1)
    graph2 = Data(x=x2, edge_index=edge_index2)

    # Create a Batch object
    batch = Batch.from_data_list([graph1, graph2])
    return batch



if __name__ == '__main__':
    # MaxWeightUpdateNN_test()
    #TestMaxWeightConv()
    #TestMaxWeightGNN()
    #TestMaxWeightGNNTensorDictModule()
    stacked1, stacked2 = NonHomogenousStackingDemo()
    batch = BatchedGraphFromTDsTest()

    #gnn_module = MaxWeightGNNTensorDictModule(in_keys=["graph"], out_keys=['action'])

    # x1 = torch.Tensor([2, 10]) #features for two node graph
    # x2 = torch.Tensor([1, 5, 3]) #features for three node graph
    #
    # edge_index1 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
    #                             [1, 0],  # edge 1 connects node 1 to node 0
    #                             ], dtype=torch.long).t().contiguous()
    # #
    # edge_index2 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
    #                             [1, 0],  # edge 1 connects node 1 to node 0
    #                             [1, 2],  # edge 2 connects node 1 to node 2
    #                             [2, 1],  # edge 3 connects node 2 to node 1
    #                             ], dtype=torch.long).t().contiguous()
    # #
    # # td_graph1 = TensorDict({'x': x1.unsqueeze(0),
    # #                         'edge_index': edge_index1.unsqueeze(0)},
    # #                         batch_size=1)
    # #
    # # td_graph2 = TensorDict({'x': x2.unsqueeze(0),
    # #                         'edge_index': edge_index2.unsqueeze(0)},
    # #                         batch_size=1)
    # #
    # # td_graphs = torch.stack([td_graph1, td_graph2], 0).contiguous()
    #
    # t1 = torch.Tensor([2, 10])
    # t2 = torch.Tensor([1, 5, 3])
    #
    # td1 = TensorDict({'x': t1, 'edge_index': edge_index1}, [])
    # td2 = TensorDict({'x': t2, 'edge_index': edge_index2}, [])
    #
    # td = torch.stack([td1, td2], 0)
    #
    # dataloader = stacked_tensor_dict_to_data(td)
    # sample = next(iter(dataloader))

    '''We are going to have to work with nestedtensors'''


    # gdata1 = GData(x1, edge_index1)

    # td1 = TensorDict({'graph': graph_td}, batch_size=1)
    # out_td1 = gnn_module(td1.copy())

    if False:
        q2 = torch.Tensor([1, 5])
        y2 = torch.Tensor([2, 1])
        x2 = torch.multiply(q2, y2)
        edge_index2 = torch.tensor([[0, 1],  # edge 0 connects node 0 to node 1
                                    [1, 0],  # edge 1 connects node 1 to node 0
                                    ], dtype=torch.long).t().contiguous()

        graph_td2 = TensorDict({'x': x2.unsqueeze(0),
                                'edge_index': edge_index2.unsqueeze(0)},
                               batch_size=1)


        td2 = TensorDict({'graph': graph_td2}, batch_size=1)

        # merge the two tensor dicts
        stacked = torch.stack([td1, td2], 0).contiguous()

        td = gnn_module(stacked)

    # print("Testing the MaxWeightGNNTensorDictModule class")
    # print(f"Input node features: {x}")
    # print(f"Input edge index: {edge_index}")
    # print(f"Output node features: {td['action']}")

