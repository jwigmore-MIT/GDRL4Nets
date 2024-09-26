import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch_geometric as pyg
import numpy as np
from modules.torchrl_development.envs.env_creation import make_env_cgs
from torch_geometric.utils import dense_to_sparse
import torch
from torchrl.envs import ExplorationType

from torchrl.envs.utils import check_env_specs
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage, ListStorage, ReplayBuffer, LazyTensorStorage
from modules.torchrl_development.envs.custom_transforms import PyGObservationTransform

from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import Sequential, GCNConv
from modules.torchrl_development.agents.cgs_agents import IndependentBernoulli
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor, ValueOperator, ActorCriticWrapper
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def visualize_graph(G, color):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_color=color, cmap="Set2")
    plt.show()


def visualize_embedding(h, color, epoch=None, loss=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    h = h.detach().cpu().numpy()
    plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
    if epoch is not None and loss is not None:
        plt.xlabel(f'Epoch: {epoch}, Loss: {loss.item():.4f}', fontsize=16)
    plt.show()


class GCN_Actor(TensorDictModule):

    def __init__(self):
        model = Sequential('x, edge_index', [
            (GCNConv(2, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            (GCNConv(64, 64), 'x, edge_index -> x'),
            ReLU(inplace=True),
            Linear(64, 1),
            Sigmoid()
        ])
        super(GCN_Actor, self).__init__(module = model, in_keys=["observation", "adj_sparse"], out_keys=["probs"])


    def forward(self, tensordict):
        data = Data(x=tensordict["observation"], edge_index=tensordict["adj_sparse"])
        tensordict["probs"] = self.module(data.x, data.edge_index).squeeze(-1)
        return tensordict

class GNN_Actor(TensorDictModule):

    def __init__(self, module, x_key = "observation", edge_index_key = "adj_sparse", out_key = "probs"):
        super(GNN_Actor, self).__init__(module = module, in_keys=[x_key, edge_index_key], out_keys=[out_key])
        self.x_key = x_key
        self.edge_index_key = edge_index_key
        self.out_key = out_key
    def forward(self, tensordict):
        tensordict[self.out_key] = self.module(Data(x=tensordict[self.x], edge_index=tensordict[self.edge_index_key])).squeeze(-1)
        return tensordict

# intialize environment
adj = np.array([[0,1,0,0,], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
arrival_dist = "Bernoulli"
arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
service_dist = "Fixed"
service_rate = np.array([1, 1, 1, 1])
env_params = {"adj": adj, "arrival_rate": arrival_rate, "service_rate": service_rate, "arrival_dist": "Bernoulli", "service_dist": "Fixed"}
make_env_keywords = {"observation_keys": ["q", "s"], "stack_observation": True, "pyg_observation": False}
env = make_env_cgs(env_params, **make_env_keywords)
# check_env_specs(env)


td = env.reset()
actor = GCN_Actor()

actor = ProbabilisticActor(
    actor,
    in_keys=["probs"],
    distribution_class=IndependentBernoulli,
    spec = env.action_spec,
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM
)

td = env.rollout(max_steps = 100, policy = actor)


#
# collector = SyncDataCollector(
#     create_env_fn=lambda: make_env_cgs(env_params, **make_env_keywords),
#     frames_per_batch=100,
#     total_frames=1000,
#     device = "cpu",
#     max_frames_per_traj=100,
#     split_trajs=True,
#     reset_when_done=True,
# )
# sampler = SamplerWithoutReplacement()
#
# data_buffer = ReplayBuffer(
#     storage = LazyTensorStorage(100),
#     batch_size=10,
#     sampler = sampler,
#     # collate_fn=lambda x: x
#     transform= PyGObservationTransform(in_keys=["observation", "adj_sparse"], out_key=["pyg_observation"])
# )
#
# for i, data in enumerate(collector):
#     data_reshape = data.reshape(-1)
#     data_buffer.extend(data_reshape)
#     for k, minibatch in enumerate(data_buffer):
#         print(k)