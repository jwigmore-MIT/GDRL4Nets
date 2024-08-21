
TorchRL is not natively compatible with Pytorch Geometric (PyG).

TorchRL is built around the TensorDict object.

A TensorDict object is a dictionary that contains Pytorch tensors.

A TensorDict is composed of a dictionary where all values are Pytorch tensors, and all tensors have the same first dimension.
This first dimension is the batch dimension.

In general, a TensorDict looks something like this:

'''
    td = TensorDict({
        'key1': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'key2': torch.tensor([[7, 8, 9], [10, 11, 12]])
    }, batch_size = 2)

'''
We can see each key has a tensor with the same first dimension (2), and then its data is in the second dimension.

We can also have keys with different dimensions, but they must have the same first dimension.

'''
    td = TensorDict({
        'key1': torch.tensor([[1, 2, 3], [4, 5, 6]]),
        'key2': torch.tensor([[[7, 8]], [[9, 10]]])
    }, batch_size = 2)

'''
Here the tensors for key1 has shape (2, 3) and the tensor for key2 has shape (2, 1, 2).

PyG is built around the Data object, where a single Data object represents a single homogenous graph.
A Data object can hold node level, link level, and graph level attributes.

For a graph with only node level attributes, we want to populate the node level attributes Data.x with a tensor of shape (num_nodes, num_node_features).
Additionally, we need to specify connectivity information in Data.edge_index, which is a tensor of shape (2, num_edges) where each column represents an edge between two nodes.

This will look something like this:

'''
    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    edge_index = torch.tensor([[0, 1], 
                               [1, 0]], dtype = torch.long).t().contiguous()
    data = Data(x = x, edge_index = edge_index)

'''
Note that to specify edge indices as a list of tuples, we need to transpose it because its supposed to have shape (2, num_edges) and not (num_edges, 2).


Now we can't store a Data object in a TensorDict object because the Data object is not a Pytorch tensor.
However, its attributes are all tensors. So we can store the attributes of a Data object in a TensorDict object. To convert a Data object to a TensorDict object, we can do the following:

'''
    td = TensorDict({key: data[key] for key in data.keys()}, [])
    td = TensorDict({key: data[key].unsqueeze(0) for key in data.keys()}, batch_size = 1)
'''
which produces a TensorDict object with the same attributes as the Data object. The first version has no batch_size,
while the second version has a batch_size of 1
    - NOTE: It looks like both versions are able to stack in a LazyStackedTensorDict object, so the batch_size parameter is not necessary.

Often we want to work with larger batches of data.  In our case this means working with multiple graphs.  
These graphs can have the same topology and dimensionality of the feature, but do not need to be.

TensorDict objects are not designed to handle this.  Normally if we had a batch_size of 2, we could stacked tensordicts together like this:
'''
td1 = (TensorDict({
        'key1': torch.tensor([[1, 2, 3]]),
        'key2': torch.tensor([[7, 8, 9]])
    }, batch_size = 1))
td2 = (TensorDict({
        'key1': torch.tensor([[4, 5, 6]]),
        'key2': torch.tensor([[10, 11, 12]])
    }, batch_size = 1))

stacked_td = torch.stack([td1, td2]).contiguous()
'''
The result would be a new TensorDict object with a batch_size of 2.  However, this is not possible if the tensors have different shapes,
or if there are different keys in the two TensorDict objects.
For example
'''
td1 = (TensorDict({
        'key1': torch.tensor([[1, 2, 3]]),
        'key2': torch.tensor([[7, 8, 9]])
    }, batch_size = 1))
td2 = (TensorDict({
        'key1': torch.tensor([[4, 5, 3]]),
        'key3': torch.tensor([[10, 11, 12]])
    }, batch_size = 1))

td3 = (TensorDict({
        'key1': torch.tensor([[4, 5]]),
        'key2': torch.tensor([[10, 11, 12]])
    }, batch_size = 1))
stacked_td12 = torch.stack([td1, td2]).contiguous() # would work but would be missing key2 and key3
stacked_td13 = torch.stack([td1, td3]).contiguous() # would not work because key1 has different shapes
'''

Instead we have to work with LazyStackedTensorDict objects. 
We create these by calling torch.stack() but omit the call to .contiguous().  

This allows us to store multiple TensorDict objects that contain the attribuites for multiple graphs.

For batch processing of graph data, we can use the Batch object from PyG. 
Batch objects are used to represent a batch of graphs as a single disconnected graph.
This allows us to perform batch processing on multiple graphs with different sizes and structures, by 
passing this disconnected graph to a Graph Neural Network (GNN) model. 
Because the graph is disconnected, message passing will not occur between the disconntected subgraphs,
allowing for batch processing of multiple graphs.

So the operations we need are:
1. TensorDict -> Data : Convert a TensorDict object to a Data object
2. Data -> TensorDict : Convert a Data object to a TensorDict object
3. LazyStackedTensorDict -> Batch: Convert a LazyStackedTensorDict object to a Batch object
4. Batch -> LazyStackedTensorDict: Convert a Batch object to a LazyStackedTensorDict object

We can use the following functions in conversions.py to perform these operations:

'''
def tensor_dict_to_data(td):
    return Data(**{key: td[key] for key in td.keys()})

def data_to_tensor_dict(data):
    return TensorDict({key: data[key] for key in data.keys()}, [])

def lazy_stacked_tensor_dict_to_batch(lstd):
    return Batch.from_data_list([tensor_dict_to_data(lstd[i]) for i in range(lstd.batch_size[0])])

def batch_to_lazy_stacked_tensor_dict(batch):
    return LazyStackedTensorDict(*[TensorDict({key: batch[i][key] for key in batch[0].keys()}, [])
                                  for i in range(batch.num_graphs)], stack_dim=0)
'''


Not only are datatypes different, but the way neural networks are handled are also different. 
Because TorchRL is supposed to work with TensorDict Objects, typically neural networks are 
wrapped by a TensorDictModule. 

A TensorDictModule is a wrapper around a Pytorch Module that allows us to pass a TensorDict object to the forward method of the Pytorch Module.

Meanwhile, PyG neural netowrks inherit from torch.nn.Module and are not designed to work with TensorDict objects.
Their forward methods expect some elements of the Data object to be passed as arguments, e.g. data.x, data.edge_index, etc.

The main question is how is our data going to be structured when using for RL? 
Environments must return TensorDict objects, but can these tensordict objects contain Data objects?




To convert a PyG neural network to a TensorDictModule, we can do the following:

'''
class PyGTensorDictModule(TensorDictModule):
    
    def __init__(self, pyg_module):
        super().__init__()
        self.pyg_module = pyg_module

    def forward(self, td):
        data = data_to_tensor_dict(td)
        return self.pyg_module(data.x, data.edge_index)
'''

''''




