from tensordict.nn import (
    TensorDictModule,
)
from tensordict import TensorDict
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
from torch_geometric.data import Batch
import torch

class BackPressureGNN(torch.nn.Module):
    def __init__(self, weight = torch.tensor([[1,-1]])):
        super(BackPressureGNN, self).__init__()
        self.input_weight = torch.nn.Parameter(weight.float())

    def forward(self, x, edge_index):
        x2 = x@self.input_weight.T

        return x2

class BackpressureGNN_Actor(TensorDictModule):

    def __init__(self,
                 feature_key="X",
                 edge_index_key = "edge_index",
                 class_edge_index_key = "class_edge_index",
                 out_keys = ["logits", "probs"],
                 init_weight = torch.tensor([[1,-1]])):
        super(BackpressureGNN_Actor, self).__init__(module = BackPressureGNN(init_weight), in_keys=[feature_key, edge_index_key, class_edge_index_key], out_keys=out_keys)

        self.feature_key = feature_key
        self.edge_index_key = edge_index_key
        self.class_edge_index_key = class_edge_index_key
        self.small_logits = torch.Tensor([0.001])

    def forward(self, input):
        if isinstance(input, TensorDict) and isinstance(input["X"], Batch): # Probabilistic actor automatically converts input to a TensorDict
            input = input["X"]
        if isinstance(input, TensorDict):
            K = input["Q"].shape[-1]
            if input[self.feature_key].dim() < 3: # < 3 # batch size is 1, meaning $\tilde X$ has shape [NK,F] an
                logits = self.module(input[self.feature_key],
                                     input[self.edge_index_key],
                                     )
                logits = logits.reshape(2, -1).T
                logits = torch.cat((self.small_logits.repeat(logits.shape[0], 1), logits), dim=1)
                probs = torch.softmax(logits, dim=-1)
                input[self.out_keys[0]] = logits.squeeze(-1)
                input[self.out_keys[1]] = probs.squeeze(-1)
            else:
                batch_graph = tensors_to_batch(input[self.feature_key], input[self.edge_index_key], input[self.class_edge_index_key], K = K)
                logits = self.module(batch_graph.x, batch_graph.edge_index)
                logits = logits.reshape(batch_graph.batch_size, 2, -1).transpose(1, 2)
                input[self.out_keys[0]] = torch.cat((self.small_logits.expand(logits.shape[0], logits.shape[1], 1), logits), dim=-1)
                input[self.out_keys[1]] = torch.softmax(input[self.out_keys[0]], dim=-1)
            return input
        elif isinstance(input, Batch):
            logits = self.module(input.x, input.edge_index)
            logits = logits.reshape(input.batch_size, 2,-1).transpose(1,2)
            logits = torch.cat((self.small_logits.expand(logits.shape[0],logits.shape[1],1), logits), dim = -1)
            # probs = torch.softmax(logits, dim = -1)
            return logits  #, probs



"""
Non-GNN Based
"""
class BackpressureActor(TensorDictModule):
    """
    All inclusive Backpressure Agent
    NOTE: The actor must know
        1. Net.M
        2. Net.K
        3. Net.link_info
    :param TensorDictModule:
    :return:
    """
    def __init__(self, net, in_keys = ["Q", "cap", "mask"], out_keys = ["action"],):
        super().__init__(module= backpressure, in_keys = in_keys, out_keys=out_keys)
        self.set_topology(net)


    def set_topology(self, net):
        self.link_info = net.link_info
        self.M = net.M
        self.K = net.K

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["cap"], td["mask"],
                                   self.M, self.K, self.link_info)
        return td


def backpressure(Q: torch.Tensor, cap: torch.Tensor, mask: torch.Tensor, M: int, K: int, link_info: dict):
    """
    Runs the backpressure algorithm given the network

    Backpressure algorithm:
        For each link:
            1. Find the class which has the greatest difference between the queue length at the start and end nodes THAT IS NOT MASKED
            2. Send the largest class using the full link capacity for each link if there is a positive differential between start and end nodes
            3. If there is no positive differential, send no packets i.e. a_i[0,1:] = 0, a_i[0,0] = Y[i]

    :param net: MultiClassMultiHop object
    :param td: TensorDict object, should contain "mask" which is a torch.Tensor of shape [M, K]

    :return: action: torch.Tensor of shape [M, K] where M is the number of links and K is the number of classes
    """

    # send the largest class using the full link capacity for each link
    action = torch.zeros([M, K])
    for m in range(M):
        diff = (Q[link_info[m]["start"]]-Q[link_info[m]["end"]])*mask[m][1:]  # mask out the classes that are not allowed to be sent
        max_class = torch.argmax(diff)
        if diff[max_class] > 0:
            action[m, max_class] = cap[m]
    return action