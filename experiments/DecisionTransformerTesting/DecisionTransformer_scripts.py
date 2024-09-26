# import
import numpy as np
import torch
import os
import wandb
from copy import deepcopy
import time
import tqdm
import sys

from tensordict import TensorDict
from torchrl.envs.utils import check_env_specs
import matplotlib.pyplot as plt

from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss

from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.agents.cgs_agents import create_mlp_actor_critic, GNN_ActorTensorDictModule
from modules.torchrl_development.envs.ConflictGraphScheduling import ConflictGraphScheduling, compute_valid_actions
from modules.torchrl_development.baseline_policies.maxweight import CGSMaxWeightActor
from modules.torchrl_development.envs.env_creation import make_env_cgs, EnvGenerator
from modules.torchrl_development.utils.metrics import compute_lta
from experiment_utils import evaluate_agent
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator, ActorCriticWrapper
from tensordict.nn import TensorDictModule
from modules.torchrl_development.agents.cgs_agents import IndependentBernoulli, GNN_TensorDictModule, tensors_to_batch


from torch_geometric.data import Data
from torch_geometric.nn import Sequential, GCNConv, SAGEConv
from torch.nn import Linear, ReLU, Sigmoid
from torch_geometric.nn import global_add_pool
from torch_geometric.data import Batch

import pickle

import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from policy_modules import *

from torchrl.objectives.value.functional import generalized_advantage_estimate

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"

def plot_data(plots, suptitle=""):
    num_plots = len(plots)
    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
    if num_plots == 1:
        axes = [axes]


    for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        if title == "MaxWeightNetwork Weights":
            data = torch.stack(data).squeeze().detach().numpy()
            print("Weights shape: ", data.shape)
            for j in range(data.shape[1]):
                if j == 0:
                    continue
                ax.plot(data[:, j], label=f"W{j}")
            ax.legend()
        else:
            ax.plot(data)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ax.set_xlabel("Minibatch")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()


def supervised_train(module, replay_buffer, num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.BCELoss, weight_decay = 0.0, lr_decay = False, reduce_on_plateau = False,
                to_plot = ["all_losses"], suptitle = "",all_losses = None, all_lrs = None, all_weights = None):
    # loss_fn = loss_fn(reduction = "none")
    loss_fn = nn.MSELoss(reduction = "none")
    # optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)

    pbar = tqdm.tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    if all_losses is None:
        all_losses = []
    if all_lrs is None:
        all_lrs = []
    if all_weights is None:
        all_weights = []
    last_n_losses = []
    for epoch in pbar:
        # add learning rate decay
        if not reduce_on_plateau and lr_decay:
            alpha = 1 - (epoch / num_training_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr * alpha

        for mb, td in enumerate(replay_buffer):
            optimizer.zero_grad()
            td = module(td)
            all_loss = loss_fn(td['probs'], td["target_action"].float())
            loss = all_loss.mean()
            loss.backward(retain_graph = False)
            optimizer.step()
            all_losses.append(loss.detach().item())
            all_lrs.append(optimizer.param_groups[0]["lr"])
            if module.get_policy_operator().module[0].module.__str__() == "MaxWeightNetwork()":
                actor_weights = module.get_policy_operator().module[0].module.get_weights()
                all_weights.append(actor_weights)
            if reduce_on_plateau:
                scheduler.step(loss)
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.detach().item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
    if len(to_plot) > 0:
        # check if all_weights is empty
        # def plot_data(plots, suptitle=""):
        #     num_plots = len(plots)
        #     fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
        #     if num_plots == 1:
        #         axes = [axes]
        #
        #
        #     if all_weights is not None:
        #         plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        #
        #     for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        #         if title == "MaxWeightNetwork Weights":
        #             data = torch.stack(data).squeeze().detach().numpy()
        #             print("Weights shape: ", data.shape)
        #             for j in range(data.shape[1]):
        #                 if j == 0:
        #                     continue
        #                 ax.plot(data[:, j], label=f"W{j}")
        #             ax.legend()
        #         else:
        #             ax.plot(data)
        #         ax.set_ylabel(ylabel)
        #         ax.set_title(title)
        #
        #     if num_plots == 2:
        #         ax[1].set_xlabel("Minibatch")
        #
        #     fig.suptitle(suptitle)
        #     fig.tight_layout()
        #     fig.show()

        plots = []
        if "all_losses" in to_plot:
            plots.append((all_losses, "Loss", "Training Loss"))
        if "all_lrs" in to_plot:
            plots.append((all_lrs, "Learning Rate", "Learning Rate Schedule"))
        if "all_weights" in to_plot:
            plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        plot_data(plots, suptitle=suptitle)
    return all_losses, all_lrs, all_weights
        # stop training if loss converges

from torch_geometric.utils import add_self_loops, scatter
def argmax_pool_neighbor_x(
    data: Data,
    flow = 'source_to_target',
) -> Data:
    r"""Max pools neighboring node features, where each feature in
    :obj:`data.x` is replaced by the feature value with the maximum value from
    the central node and its neighbors.
    """
    x, edge_index = data.x, data.edge_index

    edge_index, _ = add_self_loops(edge_index, num_nodes=data.num_nodes)

    row, col = edge_index
    row, col = (row, col) if flow == 'source_to_target' else (col, row)

    new_x = scatter(x[row], col, dim=0, dim_size=data.num_nodes, reduce='amax')
    return data

def eval_agent(agent, env_generator, max_steps, rollouts = 1, cat = True):
    agent.eval()
    tds = []
    for r in range(rollouts):
        env = env_generator.sample()
        with torch.no_grad() and set_exploration_type(ExplorationType.DETERMINISTIC):
            td = env.rollout(max_steps = max_steps, policy = agent)
            td["rollout_id"] =torch.ones_like(td)*r
            tds.append(td)
    if cat:
        tds = TensorDict.cat(tds)
    agent.train()
    return tds

def create_action_map(tensors: list, keys: list):
    ### Create a pandas dataframe that maps each observation to an action
    import pandas as pd
    df = pd.DataFrame()
    for tensor, key in zip(tensors, keys):
        temp_df = pd.DataFrame(tensor.numpy(), columns = [f"{key}{i}" for i in range(tensor.shape[1])])
        df = pd.concat([df, temp_df], axis = 1)
    # df = pd.DataFrame(observations.numpy(), columns = [f"q{i}" for i in range(observations.shape[1])])
    # df2 = pd.DataFrame(actions.numpy(), columns = [f"a{i}" for i in range(actions.shape[1])])
    # df = pd.concat([df, df2], axis = 1)
    # remove non-unique rows
    df = df.drop_duplicates()
    # sort by q values
    df = df.sort_values(by = [f"q{i}" for i in range(tensors[0].shape[1])])
    return df

def create_training_dataset(env, agent, q_max = 10):
    # get the number of nodes
    num_nodes = env.observation_spec["observation"].shape[0]
    # enumerate all possible queue states from 0 to q_max for the set of nodes
    # i.e for 3 nodes (0,0,0), (0,0,1)..., (0,0,q_max), (0,1,0), (0,1,1), ..., (q_max, q_max, q_max)
    queue_states = torch.stack(torch.meshgrid([torch.arange(q_max+1) for i in range(num_nodes)])).view(num_nodes, -1).T
    # Get the agents action for each possible queue state
    tds = []
    for i, q in enumerate(queue_states):
        td = env.reset()
        td["q"] = q
        td = env.transform(td)
        td = agent(td)
        tds.append(td)
    tds = TensorDict.stack(tds)
    return tds



"""
TRAINING PARAMETERS
"""
lr = 0.01
minibatches =10
num_training_epochs = 30
lr_decay = True

new_maxweight_data = True
max_weight_data_type = "rollout" # "rollout" or "enumerate"
train_gnn = True
train_mlp = False
""" 
ENVIRONMENT PARAMETERS
"""

## Two Set of Two Nodes
# adj = np.array([[0,1,0,0], [1,0,1,0], [0,1,0,1],[0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.3, 0.3, 0.3, 0.3])
# service_dist = "Fixed"
# service_rate = np.array([1, 1, 1, 1])


# Two Nodes
adj = np.array([[0,1], [1,0]])
arrival_dist = "Bernoulli"
arrival_rate = np.array([0.4, 0.4])
service_dist = "Fixed"
service_rate = np.array([1, 1])

## 4 Node Line Graph
# adj = np.array([[0,1,0,0], [1,0,1,0], [0,1,0,1],[0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
# service_dist = "Fixed"
# service_rate = np.array([1, 1, 1, 1])

## 8 Node Line Graph
# adj = np.array([[0,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0],[0,0,1,0,1,0,0,0],
#                 [0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
# service_dist = "Fixed"
# service_rate = np.array([1, 1, 1, 1, 1, 1, 1, 1])


interference_penalty = 0.25
reset_penalty = 100

env_params = {
    "adj": adj,
    "arrival_dist": arrival_dist,
    "arrival_rate": arrival_rate,
    "service_dist": service_dist,
    "service_rate": service_rate,
    "env_type": "CGS",
    "interference_penalty": interference_penalty,
    "reset_penalty": reset_penalty,
    "node_priority": "increasing",
}

cfg = load_config(os.path.join(SCRIPT_PATH, 'config', 'CGS_GNN_PPO_settings.yaml'))

cfg.training_make_env_kwargs.observation_keys.append("node_priority") # required to differentiate between nodes with the same output embedding

gnn_env_generator = EnvGenerator(input_params=env_params,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             cgs = True)


env = gnn_env_generator.sample()

check_env_specs(env)


"""
CREATE GNN ACTOR AND CRITIC ARCHITECTURES
"""
node_features = env.observation_spec["observation"].shape[-1]

from transformers import GraphormerModel, GraphormerConfig



