from torch_geometric.nn import SAGEConv
import os
from modules.torchrl_development.envs.custom_transforms import MCMHPygTransform, SymLogTransform, MCMHPygQTransform
from torchrl.envs.transforms import TransformedEnv
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from torchrl.envs.utils import check_env_specs
import torch

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"
env_name= "env3"


"""
Get Environment
"""
file_path = f"../envs/{env_name}.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
base_env = MultiClassMultiHop(**env_info)
link_env = TransformedEnv(base_env, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=base_env))
check_env_specs(link_env)
td = link_env.rollout(max_steps = 10)[-1]
td["Q"] = torch.arange(0, link_env.N*link_env.K).reshape(link_env.N, link_env.K).float()
W = torch.tensor([[1,-1]]).float()

Xp = td["X"]@W.T

from torch_geometric.utils import spmm
from torch_geometric.transforms import ToSparseTensor, to_sparse_tensor
from torch_geometric.utils import to_torch_coo_tensor

# this will find the max of all class neighbors for each link. This could be used as a message passing function to
class_edge_index_sparse  = to_torch_coo_tensor(td["class_edge_index"])
test = spmm(class_edge_index_sparse.T, Xp, reduce='max')
test2 = test.view(-1, link_env.K)


# Instead we can just perfom the reshape operation on the Xp tensor
Xp2 = Xp.view(-1, link_env.K)

# Action = one hot encoding of the class with the max queue length
action = torch.argmax(Xp2, dim=1)
action_one_hot = torch.nn.functional.one_hot(action, num_classes=link_env.K)

# To incorporate the idle action, we would add a column of small values to Xp2 Tensor
Xp2 = torch.cat([0.1*torch.ones(Xp2.shape[0], 1), Xp2 ], dim=1)
action = torch.argmax(Xp2*td["mask"], dim=1)
action_one_hot = torch.nn.functional.one_hot(action, num_classes=link_env.K+1)



class BackPressureGNN(torch.nn.Module):
    def __init__(self, weight = torch.tensor([[1,-1]])):
        super(BackPressureGNN, self).__init__()
        self.input_weight = torch.nn.Parameter(weight.float())

    def forward(self, x, edge_index):
        x2 = x@self.input_weight.T

        return x2

from tensordict.nn import (
    TensorDictModule,
)
from tensordict import TensorDict
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
from torch_geometric.data import Batch
class BackpressureGNN_Actor(TensorDictModule):

    def __init__(self,
                 feature_key="X",
                 edge_index_key = "edge_index",
                 class_edge_index_key = "class_edge_index",
                 out_keys = ["logits"]):
        super(BackpressureGNN_Actor, self).__init__(module = BackPressureGNN(), in_keys=[feature_key, edge_index_key, class_edge_index_key], out_keys=out_keys)

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
                # input[self.out_keys[1]] = probs.squeeze(-1)
            else:
                batch_graph = tensors_to_batch(input[self.feature_key], input[self.edge_index_key], input[self.class_edge_index_key], K = K)
                logits = self.module(batch_graph.x, batch_graph.edge_index)
                logits = logits.reshape(batch_graph.batch_size, 2, -1).transpose(1, 2)
                input[self.out_keys[0]] = torch.cat((self.small_logits.expand(logits.shape[0], logits.shape[1], 1), logits), dim=-1)
            return input
        elif isinstance(input, Batch):
            logits = self.module(input.x, input.edge_index)
            logits = logits.reshape(input.batch_size, 2,-1).transpose(1,2)
            logits = torch.cat((self.small_logits.expand(logits.shape[0],logits.shape[1],1), logits), dim = -1)
            # probs = torch.softmax(logits, dim = -1)
            return logits  #, probs




# Test Backpressure GNN
from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical
from torchrl.modules import ProbabilisticActor
from torchrl.envs import ExplorationType, set_exploration_type
from modules.torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
from experiments.MultiClassMultiHopDevelopment.development.backpressure import BackpressureActor
bp_actor = BackpressureActor(net=link_env)

bp_gnn_actor = BackpressureGNN_Actor()
bp_gnn_actor = ProbabilisticActor(bp_gnn_actor,
                            in_keys = ["logits", "mask"],
                            distribution_class= MaskedOneHotCategorical,
                            spec = link_env.action_spec,
                            default_interaction_type = ExplorationType.MODE,
                            return_log_prob=True,
                             )
td = link_env.rollout(max_steps=1000, policy=bp_gnn_actor)
bp_actions = torch.zeros_like(td["action"])
for i in range(len(td)):
    bp_action = bp_actor(td[i])["action"]
    # modify bp_action to add another column which value = 1 if all elements of the row = 0
    idle = (bp_action == 0).all(axis=-1).unsqueeze(-1).float()
    bp_action = torch.cat([idle, bp_action], dim=-1)
    bp_actions[i] = bp_action
same = (bp_actions == td["action"]).all()
print("Are the actions the same?", same)

# plot
lta = compute_lta(-td["next","reward"])
fig, ax = plt.subplots()
ax.plot(lta)
ax.set_xlabel("Time")
ax.set_ylabel("Queue Length")
ax.legend()
ax.set_title("Backpressure GNN")
fig.show()



td2 = link_env.rollout(max_steps=1000, policy=bp_actor)
lta = compute_lta(-td2["next","reward"])
fig, ax = plt.subplots()
ax.plot(lta)
ax.set_xlabel("Time")
ax.set_ylabel("Queue Length")
ax.legend()
ax.set_title("Backpressure Actor")
fig.show()

# Now lets test if forward pass works with a batch of data
td5 = td[0:5]

batch = tensors_to_batch(td5["X"], td5["edge_index"], td5["class_edge_index"], K = td5["Q"].shape[-1])
logits =  bp_gnn_actor.module[0](batch)
print(f"Checking if done correctly: {(logits == td5['logits']).all()}")

# we would then need to reshape in order to then

# Now lets test if forward pass works with a batch of data as a tensordict
td5b = bp_gnn_actor(td5)
print(f"Checking if done correctly: {(logits == td5b['logits']).all()}")
