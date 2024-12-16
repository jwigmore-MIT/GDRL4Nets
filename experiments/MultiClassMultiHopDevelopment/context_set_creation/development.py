
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt





file_path = "../envs/grid_3x3.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)


net = MultiClassMultiHop(**env_info)

# plot net.graphx with node and edge labels
G = net.graphx
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()
td = net.reset()




