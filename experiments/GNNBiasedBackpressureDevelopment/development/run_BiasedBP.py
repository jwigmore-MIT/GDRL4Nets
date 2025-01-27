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


file_path = "../envs/grid_5x5.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
net = MultiClassMultiHopBP(**env_info)
weighted = False
alpha = 0
bias, link_sp_dist = create_sp_bias(net,weighted, alpha)
net.set_bias(bias)

#plot the graph
G = net.graphx
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()


start_time = time.time()
td = net.rollout(3000)
end_time = time.time()
rollout_time = end_time-start_time
print(f"Rollout Time: {rollout_time:.2f} seconds")
y_lim = 7

fig, ax = plt.subplots()
plt.plot(compute_lta(td["Q"].sum((1,2))))
fig.suptitle(f"Total Backlog; alpha = {alpha}")
fig.show()

# fig, ax = plt.subplots()
# plt.plot(td["backlog"])
# fig.suptitle("Total Backlog")
# fig.show()