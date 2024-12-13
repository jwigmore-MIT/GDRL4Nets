
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
env = TransformedEnv(base_env, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=base_env, include = ["sp_dist"]))

td = env.rollout(max_steps = 10)

sp_dist = env.sp_dist
edge_index = env.edge_index # Coordinate (COO) Format [2,M]
end_nodes = edge_index[1] # end nodes