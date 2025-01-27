import torch
import torch.nn as nn
import torch.nn.functional as torchf
from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import SDPA_layer, NodeAttentionConv, DeeperNodeAttentionGNN
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
import warnings
from torch.nn import Sigmoid

warnings.filterwarnings("ignore", category=DeprecationWarning)


N = 5
K = 3
F = 2
D  = 16


X = torch.rand((N,K,F))
""" Manual Version
embed_mlp = nn.Linear(F, D)
X_embed = embed_mlp(X)

# Create QKV MLP
qkv_mlp = nn.Linear(D, D*3, bias = False)

# Get Q, K, V
QKV = qkv_mlp(X_embed)
Q, K, V = QKV.chunk(3, dim = -1)

# Apply scaled dot product attention
H = torchf.scaled_dot_product_attention(Q, K, V)

"""


# test the SDPA layer
sdpa = SDPA_layer(X.shape[-1], D)
H = sdpa(X)

first_layer = NodeAttentionConv(in_channels= X.shape[-1],
                                out_channels=D,
                                pass_message=False,
                                )
second_layer = NodeAttentionConv(in_channels= D,
                                    out_channels=D,
                                    pass_message=True,
                                 edge_channels=1
                                    )

# Lets get an actual network
file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
td = env.get_rep(include_bias=False)


H = first_layer(td["X"])
H2 = second_layer(H, edge_index = td["edge_index"], edge_attr = td["edge_attr"])

model = DeeperNodeAttentionGNN(
    node_channels = td["X"].shape[-1],
    edge_channels = td["edge_attr"].shape[-1],
    hidden_channels = D,
    num_layers = 2,
    output_channels=1,
    output_activation=Sigmoid
)

Y = model(td["X"], td["edge_index"], td["edge_attr"])
