from experiments.GNNBiasedBackpressureDevelopment.utils import plot_nx_graph

from NetworkRunner import create_network_runner
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
import json
from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import DeeperNodeAttentionGNN
from experiments.GNNBiasedBackpressureDevelopment.modules.modules import NormalWrapper
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal
from tensordict.nn import InteractionType
# import optimizer
import torch.optim as optim
import tensordict
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
from torch_geometric.data import Data, Batch
import networkx as nx
import torch_geometric as pyg
import matplotlib.pyplot as plt
import math

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


file_path = "../envs/grid_3x3.json"
env_info = json.load(open(file_path, 'r'))
env_info["action_func"] = "bpi"
env = MultiClassMultiHopBP(**env_info)
network_specs = env.get_rep()

# Create the Model
model = DeeperNodeAttentionGNN(
    node_channels = network_specs["X"].shape[-1],
    edge_channels = network_specs["edge_attr"].shape[-1],
    hidden_channels =8,
    num_layers = 4,
    output_channels=2,
    output_activation=None,
    edge_decoder=True
)

norm_module = NormalWrapper(model)
actor = ProbabilisticActor(norm_module,
                            in_keys = ["loc", "scale"],
                            out_keys = ["bias"],
                            distribution_class=IndependentNormal,
                           distribution_kwargs={"tanh_loc": True},
                            return_log_prob=True,
                            default_interaction_type = InteractionType.RANDOM
                            )

collector = SyncDataCollector(
    create_env_fn=create_network_runner(env=env, max_steps=2000, graph=True),
    policy=actor,
    frames_per_batch=1,
    total_frames=1,
    reset_at_each_iter=True,
)

result = collector.rollout()
data = result["next","data"]
fig,ax = plot_nx_graph(data, data.loc.squeeze(), K=4, epoch = 1, reward = 1)
fig.show()
