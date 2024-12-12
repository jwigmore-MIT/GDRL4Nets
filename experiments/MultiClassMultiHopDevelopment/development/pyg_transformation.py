
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt
from torchrl.envs.transforms import ObservationTransform, TransformedEnv
from modules.torchrl_development.envs.custom_transforms import MCMHPygTransform

from tensordict import (
    TensorDictBase,
)
from torchrl.data.tensor_specs import (
    ContinuousBox,
    TensorSpec,
    UnboundedContinuousTensorSpec,
    Unbounded,
    Composite,
    NonTensor,
)


file_path = "../envs/env2.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)


net = MultiClassMultiHop(**env_info)

# plot net.graphx with node and edge labels
G = net.graphx
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()
env = TransformedEnv(net, MCMHPygTransform(in_keys=["Q"], out_keys=["X"], env=net))
td = env.reset()

G2 = nx.DiGraph(td["edge_index"].T.tolist())
pos2 = nx.fruchterman_reingold_layout(G2, iterations=1000)
nx.draw(G2, pos2, with_labels=True, font_weight='bold')
plt.show()

G3 = nx.DiGraph(td["class_edge_index"].T.tolist())
pos3 = nx.fruchterman_reingold_layout(G3, iterations=1000)
nx.draw(G3, pos3, with_labels=True, font_weight='bold')
plt.show()

td2 = env.rollout(max_steps=2)



"""
Unit tests
1. We should have net.K disconnected subgraphs in the graph induced by td["edge_index"]
2. td["edge_index"].shape should be (2, net.M*net.K)
3. We should have net.N disconnected subgraphs in the graph induced by td["class_edge_index"]
4. td["class_edge_index"].shape should be (2, net.N*net.K*(net.K-1))
5. td["X"].shape should be (net.N*net.K,3)
"""

# Unit test 1
G2 = nx.DiGraph(td["edge_index"].T.tolist())
assert nx.number_weakly_connected_components(G2) == net.K

# Unit test 2
assert td["edge_index"].shape == (2, net.M*net.K)

# Unit test 3
G3 = nx.DiGraph(td["class_edge_index"].T.tolist())
assert nx.number_weakly_connected_components(G3) == net.N

# Unit test 4
assert td["class_edge_index"].shape == (2, net.N*net.K*(net.K-1))

# Unit test 5
assert td["X"].shape == (net.N*net.K,3)


