import tensordict

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
from copy import deepcopy

from experiments.GNNBiasedBackpressureDevelopment.utils import tensors_to_batch, plot_nx_graph
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':
    # set all random seeds
    torch.manual_seed(12)


    p_envs = 2
    link_rewards = "shared"
    regress_to_best = False


    file_path = "../envs/line2.json"
    env_info = json.load(open(file_path, 'r'))
    # env_info['class_info'] = [env_info['class_info'][0]]

    env_info["action_func"] = "bpi"
    env = MultiClassMultiHopBP(**env_info)
    network_specs = env.get_rep()

    # Create the Model
    model = DeeperNodeAttentionGNN(
        node_channels = network_specs["X"].shape[-1],
        edge_channels = network_specs["edge_attr"].shape[-1],
        hidden_channels =8,
        num_layers = 4,
        output_channels=1,
        output_activation=None,
        edge_decoder=True
    )
    # TODO: Use Tanh normal wrapper instead to keep bias within -10,10
    norm_module = NormalWrapper(model, scale_bias=0.2)
    actor = ProbabilisticActor(norm_module,
                                in_keys = ["loc", "scale"],
                                out_keys = ["bias"],
                                distribution_class=IndependentNormal,
                               distribution_kwargs={},
                                return_log_prob=True,
                                default_interaction_type = ExplorationType.DETERMINISTIC
                                )

    create_runner = lambda: create_network_runner(env=env, max_steps=2000, graph=True, link_rewards=link_rewards)
    collector = SyncDataCollector(
        create_env_fn= create_runner,
        policy=actor,
        frames_per_batch=1,
        total_frames=100,
        exploration_type=ExplorationType.DETERMINISTIC

    )

    res = collector.rollout()
    res2 = collector.rollout()

    graph = res["next", "data"]

    graph_dist = actor.get_dist(
        TensorDict({"X": graph.X, "edge_index": graph.edge_index, "edge_attr": graph.edge_attr}))
