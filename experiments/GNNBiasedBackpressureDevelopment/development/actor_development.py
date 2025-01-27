

from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import SDPA_layer, NodeAttentionConv, DeeperNodeAttentionGNN
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
from torch.nn import Sigmoid
from torchrl.modules import NormalParamExtractor, IndependentNormal
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from torch_geometric.data import Data, Batch

file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
td = env.get_rep()


# Create the Model
model = DeeperNodeAttentionGNN(
    node_channels = td["X"].shape[-1],
    edge_channels = td["edge_attr"].shape[-1],
    hidden_channels =16,
    num_layers = 2,
    output_channels=2,
    output_activation=Sigmoid,
    edge_decoder=True
)



# model = torch.nn.Sequential(
#     model,
#     NormalParamExtractor()
# )
from tensordict import TensorDict, NonTensorData
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor
class DNAGNNNormalWrapper(TensorDictModule):
    def __init__(self, module):
        super(DNAGNNNormalWrapper, self).__init__(module = module, in_keys=["X", "edge_index", "edge_attr"], out_keys=["loc", "scale"])
        self.normal_params = NormalParamExtractor()

    def forward(self, td):
        logits = self.module(x = td["X"],edge_index = td["edge_index"],edge_attr =  td["edge_attr"])
        td["loc"], td["scale"] = self.normal_params(logits)
        return td




td_module = DNAGNNNormalWrapper(model)
actor = ProbabilisticActor(td_module,
                            in_keys = ["loc", "scale"],
                            out_keys = ["bias"],
                            distribution_class=IndependentNormal,
                            return_log_prob=True
                            )
# Test forward pass
td = actor(td)
#
# # Create Probabilistic Actor
# from tensordict import TensorDict, NonTensorData
# from tensordict.nn import TensorDictModule
# from torchrl.modules import ProbabilisticActor
#
#
#
#
#
# data = TensorDict({"input": NonTensorData(rep_graph)})
#
# td_module = TensorDictModule(model,
#                              in_keys = ["input"],
#                              out_keys = ["loc", "scale"]
#                              )
#
# actor = ProbabilisticActor(td_module,
#                             in_keys = ["loc", "scale"],
#                             out_keys = ["bias"],
#                             distribution_class=IndependentNormal,
#                             return_log_prob=True
#                             )
#
# output = actor(data)
#
#
#






