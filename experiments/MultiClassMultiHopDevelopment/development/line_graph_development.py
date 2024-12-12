
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

env_name= "env3"
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
env = TransformedEnv(base_env, MCMHPygQTransform(in_keys=["Q"], out_keys=["X"], env=base_env))
env = TransformedEnv(env, SymLogTransform(in_keys=["X"], out_keys=["X"]))
check_env_specs(env)
env.Q = torch.arange(0, env.N*env.K).reshape(env.N, env.K).float()

edge_index = env.edge_index # Coordinate (COO) Format [2,M]
edge_list = edge_index.T # Edge list format [M,2]
Qe_0s = env.Q[edge_list[0,0]] # Queue length for all classes at start node of edge 0
Qe_0e = env.Q[edge_list[0,1]] # Queue length for all classes at start node of edge 0
# Queue length for all classes at start and end node of edge 0

# This would be [2,K] where the first row is the queue length at the start node and the second row is the queue length at the end node
Qe_0 = torch.stack([Qe_0s, Qe_0e], dim=0)

Qe_0 = Qe_0.unsqueeze(dim = 0) # (1,2,K)


# Lets do the same but for all edges
Qe = torch.stack([env.Q[edge[0]] for edge in edge_list], dim=0)

start_queues = env.Q[edge_list[:,0]]
end_queues = env.Q[edge_list[:,1]]

Qe = torch.stack([start_queues, end_queues], dim=1).transpose(-1,-2) #[M,2,K]

# I want to represent Qe as a [M*K, 2] tensor where we take the last dimension of stack them on top of eachother
Xq = Qe.reshape(-1, Qe.shape[-1]) # [M*K, 2]

# Now simplify this to a single line -> This will be in the _call_ method of the MCMHPygQTransform as
# tensordict["X"] = tensordict["Q"][self.edge_list].reshape(-1, 2)  ### For Q size only edge features
Xq2 = env.Q[edge_index].T.reshape(-1, 2) # [M*K, 2]


# Now lets create the adjancency matrix the Multiclass Link graph
# The features are for Link 0, class 0,  Link 1, class 0,...
link_list = []
for m1, edge in enumerate(edge_list):
    for m2,other_edge in enumerate(edge_list):
        if edge[1] == other_edge[0]:
            link_list.append([m1, m2])
link_list = torch.tensor(link_list)
link_index = link_list.T #adjacency matrix of link graph in COO format
N2 = env.M
M2 = link_index.shape[-1]
# Now lets create the adjancency matrix the Multiclass Link graph
for k in range(env.K):
    link_index = torch.cat([link_index, link_index[:,-M2:] + M2], dim=-1)

# We need the class edge index to pass messages between classes of the same link
class_link_index= list()
for m in range(M2):
    # get all indices
    indices = [i for i in range(m, M2 * env.K, M2)]
    # create edge index for fully connected graph of indices
    class_link_index.extend([[i, j] for i in indices for j in indices if i != j])
class_link_index = torch.tensor(class_link_index, dtype=torch.long).T


from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
base_env = MultiClassMultiHop(**env_info)
link_env = TransformedEnv(base_env, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=base_env))
check_env_specs(link_env)
td = link_env.reset()
td["Q"] = torch.arange(0, link_env.N*link_env.K).reshape(link_env.N, link_env.K).float()
td = link_env.transform._call(td)


# Backpressure Actor Developmen
