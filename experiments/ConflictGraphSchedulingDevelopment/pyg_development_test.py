import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch_geometric as pyg
import numpy as np
from modules.torchrl_development.envs.env_creation import make_env_cgs
from torch_geometric.utils import dense_to_sparse
import torch

# intialize environment
adj = np.array([[0,1,0,0,], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
arrival_dist = "Bernoulli"
arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
service_dist = "Fixed"
service_rate = np.array([1, 1, 1, 1])
env_params = {"adj": adj, "arrival_rate": arrival_rate, "service_rate": service_rate, "arrival_dist": "Bernoulli", "service_dist": "Fixed"}
make_env_keywords = {"observation_keys": ["q", "s"], "stack_observation": True}
env = make_env_cgs(env_params, **make_env_keywords)


adj_sparse, x = dense_to_sparse(torch.tensor(adj))