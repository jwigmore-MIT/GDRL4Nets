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

file_path = "../envs/grid_5x5.json"
env_info = json.load(open(file_path, 'r'))
# env_info["action_func"] = "bp+interference"
env_info["action_func"] = "bpi"
net = MultiClassMultiHopBP(**env_info)


save_path = "../comparison"

td = net.rollout(2000)
y_lim = 7
# K = net.K
# # plot the backlog for each queue in for each class
# # specify the first 4 colors in the default color cycle
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
# # for k,color in zip(range(4), colors[:K]):
# #     fig, axs = plt.subplots(ncols = int(net.N**(1/2)), nrows = int(net.N**(1/2)), figsize=(30,30))
# #     for n,ax in zip(range(net.N),axs.flatten()):
# #         ax.plot(compute_lta(td["Q"][:,n,k]), color = color)
# #         ax.set_ylim(0,y_lim)
# #         ax.set_title(f"Node {n}")
# #     # set title of the figure
# #     fig.suptitle(f"Backlog for class {k} at each node")
# #     fig.tight_layout()
# #     fig.show()
# #     # fig.savefig(os.path.join(save_path, f"{env_info['action_func']}_{k}"))
#
fig, ax = plt.subplots()
plt.plot(compute_lta(td["Q"].sum((1,2))))
fig.suptitle("Total Backlog")
fig.show()
# # fig.savefig(os.path.join(save_path, f"{env_info['action_func']}_Total"))
#
#
#
