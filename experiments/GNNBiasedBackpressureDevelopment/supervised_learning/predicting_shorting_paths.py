"""
Goal: given a network instance net.get_rep(), predict the average utilization over each link


"""


from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import SDPA_layer, NodeAttentionConv, DeeperNodeAttentionGNN
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
from torch.nn import Sigmoid
import torch
import torch.optim as optim
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


file_path = "../envs/grid_5x5.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
rep = env.get_rep(include_bias=False)

# Create the Model
model = DeeperNodeAttentionGNN(
    node_channels = rep["X"].shape[-1],
    edge_channels = rep["edge_attr"].shape[-1],
    hidden_channels =16,
    num_layers = 2,
    output_channels=1,
    output_activation=Sigmoid,
    edge_decoder=False
)

# Test forward pass
out = model(rep["X"], rep["edge_index"], rep["edge_attr"])

# Collect Rollout
td = env.rollout(1000)

actions = td["next", "action"].float()
freq = actions.mean(dim = 0).unsqueeze(-1)


# train the model to predict freq
optimizer = optim.Adam(model.parameters(), lr = 0.01)
criterion = nn.MSELoss()
for t in range(10000):
    optimizer.zero_grad()
    out = model(rep["X"], rep["edge_index"], rep["edge_attr"])
    loss = criterion(out, freq)
    loss.backward()
    optimizer.step()
    if t % 100 == 0:
        print(f"Epoch {t}, Loss: {loss.item()}")

diff = out.detach() - freq
print(f"Mean Absolute Error: {diff.abs().mean()}")

test = torch.ones(freq.shape)*freq.mean()

diff2 = test - freq
print(f"Mean Absolute Error 2: {diff2.abs().mean()}")