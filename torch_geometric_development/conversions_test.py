import torch
from torch_geometric.data import Data, Batch
from tensordict import TensorDict, LazyStackedTensorDict
from torch_geometric_development.conversions import tensor_dict_to_data, data_to_tensor_dict, lazy_stacked_tensor_dict_to_batch, batch_to_lazy_stacked_tensor_dict

def test_tensor_dict_to_data():
    td = TensorDict({'x': torch.rand(10, 10), 'edge_index': torch.randint(0, 10, (2, 50))}, [])
    data = tensor_dict_to_data(td)
    assert isinstance(data, Data)
    assert torch.all(data.x == td['x'])
    assert torch.all(data.edge_index == td['edge_index'])

def test_data_to_tensor_dict():
    data = Data(x=torch.rand(10, 10), edge_index=torch.randint(0, 10, (2, 50)))
    td = data_to_tensor_dict(data)
    assert isinstance(td, TensorDict)
    assert torch.all(td['x'] == data.x)
    assert torch.all(td['edge_index'] == data.edge_index)

def test_lazy_stacked_tensor_dict_to_batch():
    # Corresponds to graph with 10 nodes and 40 edges
    td1 = TensorDict({'x': torch.rand(3, 2), 'edge_index': torch.randint(0, 3, (2, 5))}, [])
    # Corresponds to graph with 8 nodes and 50 edges
    td2 = TensorDict({'x': torch.rand(4, 2), 'edge_index': torch.randint(0, 4, (2, 6))}, [])
    # NOTE:
    lstd = torch.stack([td1, td2],  dim = 0)
    batch = lazy_stacked_tensor_dict_to_batch(lstd)
    assert isinstance(batch, Batch)
    assert batch.num_graphs == lstd.batch_size[0]
    #assert torch.all(batch['x'] == torch.cat([lstd['x'][i] for i in range(lstd.batch_size[0])]))
def test_batch_to_lazy_stacked_tensor_dict():
    data_list = [Data(x=torch.rand(10, 10), edge_index=torch.randint(0, 10, (2, 50))) for _ in range(2)]
    batch = Batch.from_data_list(data_list)
    lstd = batch_to_lazy_stacked_tensor_dict(batch)
    assert isinstance(lstd, LazyStackedTensorDict)
    assert lstd.batch_size[0] == batch.num_graphs
    #assert torch.all(lstd['x'] == torch.stack([batch[i]['x'] for i in range(batch.num_graphs)]))
    #assert torch.all(lstd['edge_index'] == torch.stack([batch[i]['edge_index'] for i in range(batch.num_graphs)]))

if __name__ == "__main__":
    test_tensor_dict_to_data()
    test_data_to_tensor_dict()
    test_lazy_stacked_tensor_dict_to_batch()
    test_batch_to_lazy_stacked_tensor_dict()