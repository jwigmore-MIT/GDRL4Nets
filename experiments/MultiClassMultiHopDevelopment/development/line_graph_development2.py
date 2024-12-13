
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


"""
PARAMETERS
"""
gnn_layer = 1
hidden_channels = 1

lr = 0.001
minibatches =100
num_training_epochs = 30
lr_decay = True

env_name= "env2"
new_backpressure_data = False
training_data_amount = [10_000, 1]



"""
Get Environment
"""
file_path = f"../envs/{env_name}.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

# init env
base_env = MultiClassMultiHop(**env_info)

from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
env = TransformedEnv(base_env, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=base_env, include = ["sp_dist"]))
actor = BackpressureActor(env)
td = env.rollout(max_steps = 10, policy = actor)

sp_dist = env.sp_dist
edge_index = env.edge_index # Coordinate (COO) Format [2,M]
end_nodes = edge_index[1] # end nodes


# Get indices of links that share the same source node
source_nodes = edge_index[0]
unique_sources = torch.unique(source_nodes)
source_indices = []
for source in unique_sources:
    source_indices.append(torch.where(source_nodes == source)[0])

# Get the sum
Q = td["Q"][-1]
action = td["action"][-1]

Q = torch.ones_like(Q)
Qdiff = torch.zeros_like(Q)
action = torch.zeros_like(action)
action[:,0] = 1
start_Q = Q.clone()
for m in range(env.M):
    # first make sure that sum over action[m] <= capacity of link m
    assert action[m].sum(dim = 0) <= env.cap[m]
    to_transmit = torch.min(action[m], start_Q[env.start_nodes[m]])
    start_Q[env.start_nodes[m]] -= to_transmit
    Qdiff[env.start_nodes[m]] -= to_transmit
    Qdiff[env.end_nodes[m]] += to_transmit

new_Q = Q + Qdiff




# Ensure that action[source_indices[i]].sum(dim =0) < Q[i] for all i
# for i in range(len(source_indices)):
#     assert (action[source_indices[i]].sum(dim = 0) <= Q[source_indices[i]].sum(dim = 0)).all()
#
# # modify action such that action[source_indices[i]].sum(dim =0) = Q[i] for all i
# for i in range(len(source_indices)):
#     # sampling without replacement to ensure that the sum is equal to Q
#     actions = action[source_indices[i]]
#     indices = torch.randperm(len(source_indices[i]))[:int(Q[source_indices[i]].sum())]
#     action[source_indices[i]] = torch.zeros_like(action[source_indices[i]])
#     action[source_indices[i]][indices] = 1

