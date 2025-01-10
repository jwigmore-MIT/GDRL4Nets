import random
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
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

file_path = "../envs/grid_4x4_mod/grid_4x4_node_subsampled_gWscje.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
net = MultiClassMultiHopBP(**env_info)
td = net.reset()
T = 1000
total_arrivals = td["arrivals"].sum().item()
total_departures = 0
backlog = torch.zeros([T])
tds = [td]
for t in range(T):
    if t < 1:
        td = net.step(td)
    else:
        td = net.step(td["next"])
    tds.append(td["next"])
    backlog[t] = td["Q"].sum()
    total_departures += td["departures"].sum().item()
    total_arrivals += td["arrivals"].sum().item()
    assert total_arrivals == td["Q"].sum().item() + total_departures

