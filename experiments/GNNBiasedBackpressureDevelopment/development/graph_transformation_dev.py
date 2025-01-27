import random
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP, create_sp_bias
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor, BackpressureGNN_Actor
import time
import warnings
import torch
from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor
from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical
warnings.filterwarnings("ignore", category=DeprecationWarning)
from modules.torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
import os
import networkx as nx
from torch_geometric.data import Data


file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)

td = env.rollout(10)


obs = env._get_observation()

# This gets us the N,K,F where F=2 node feature tensor
X = torch.stack([obs["arrival_rates"],obs["sp_dist"]], dim = 2)

# How to convert a
edge_feature = obs["link_rates"][0]
# Transform to a K,1 tensor from a 1,1
edge_feature_matrix = edge_feature.repeat(env.K,1)

# Convert to a pyg data object
data = Data(x=X, edge_index=env.edge_index, edge_attr=obs["link_rates"].unsqueeze(-1))


# Now lets use the env function
env_rep = env.get_rep()
graph = Data(x=env_rep["X"], edge_index=env_rep["edge_index"], edge_attr=env_rep["edge_attr"])




