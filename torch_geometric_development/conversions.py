from torch_geometric.data import Data, Batch
from tensordict import TensorDict, LazyStackedTensorDict
import torch


def tensor_dict_to_data(td):
    return Data(**{key: td[key] for key in td.keys()})

def data_to_tensor_dict(data):
    return TensorDict({key: data[key] for key in data.keys()}, [])

def lazy_stacked_tensor_dict_to_batch(lstd):
    # Need to handle case where lstd contains TensorDicts with a batch size > 1
    if lstd.batch_size.__len__()== 1:
        return Batch.from_data_list([tensor_dict_to_data(lstd[i]) for i in range(lstd.batch_size[0])])
    else: # each stacked tensor dict has multiple batches (i.e. graphs with the same topology and feature dimensions)
        batches = [lazy_stacked_tensor_dict_to_batch(lstd[i]) for i in range(lstd.batch_size[0])]
        return Batch.from_data_list(batches)

def batch_to_lazy_stacked_tensor_dict(batch):
    return LazyStackedTensorDict(*[TensorDict({key: batch[i][key] for key in batch[0].keys()}, [])
                                  for i in range(batch.num_graphs)], stack_dim=0)




