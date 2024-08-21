import json
import os
from torchrl_development.utils.configuration import load_config
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.actors import create_actor_critic
import torch
from torchrl.record.loggers import get_logger
from datetime import datetime
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm
from torchrl_development.utils.metrics import compute_lta
import numpy as np
from torchrl_development.maxweight import MaxWeightActor
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torchrl_development.mdp_actors import MDP_actor, MDP_module
from MDP_Solver.SingleHopMDP import SingleHopMDP
import sys
from torchrl.data.replay_buffers import ReplayBuffer, ListStorage, LazyTensorStorage, SamplerWithoutReplacement
import tensordict
from tensordict import TensorDict

from experiments.experiment8.maxweight_comparison.CustomNNs import FeedForwardNN, LinearNetwork, MaxWeightNetwork, NN_Actor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))

def max_weight_policy(Q,Y, w = None):
    """Computes the MaxWeight policy action given the Q and Y array"""
    A = torch.zeros((Q.shape[0],Q.shape[1]+1), dtype=torch.int)
    if w is None:
        w = torch.ones(Q.shape[1])
    for i in range(Q.shape[0]):
        v = Q[i]*Y[i]*w
        if torch.all(v==0):
            A[i,0] = 1
        else:
            max_index = torch.argmax(v)
            A[i,max_index+1] = 1
    return A

def train_module(module, td, in_keys = ["Q", "Y"], num_training_epochs=1000, lr=0.001,
                 loss_fn = nn.BCEWithLogitsLoss()):
    loss_fn = loss_fn
    optimizer = Adam(module.parameters(), lr=lr)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    last_n_losses = []
    for epoch in pbar:
        optimizer.zero_grad()
        x = torch.cat([td[key] for key in in_keys], dim=1)

        A = module(x)
        loss = loss_fn(A.float(), td["action"].float())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            last_n_losses.append(loss.item())
            pbar.set_postfix({f"Epoch": epoch, f"Loss": loss.item()})
            if len(last_n_losses) > 10:
                last_n_losses.pop(0)
                if np.std(last_n_losses) < 1e-6:
                    break
        # stop training if loss converges

def train_module2(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.BCEWithLogitsLoss(), weight_decay = 1e-5):
    loss_fn = loss_fn
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    last_n_losses = []
    for epoch in pbar:

        for mb, td in enumerate(replay_buffer):
            optimizer.zero_grad()
            td["Q"] = td["Q"].float()
            td["Y"] = td["Y"].float()
            A = module(td)
            loss = loss_fn(A['action'].float(), td["true_action"].float())
            loss.backward()
            optimizer.step()
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
        # stop training if loss converges

def get_module_error_rate(module, td, inputs = ["Q", "Y"]):
    module.eval()
    actions = module(torch.cat([td[key] for key in inputs], dim=1))
    error = torch.norm(actions - td["action"].float())
    error_rate = error / td["action"].shape[0]
    return error_rate, error




if __name__ == "__main__":
    import pickle
    env_id = 0
    rollout_length = 10000
    num_rollouts = 3
    training_epochs = 100

    new_mdp_trajectories = True

    actor_depth = 2
    actor_cells = 32

    results = {}
    test_context_set_path = 'SH1_context_set.json'
    mdp_path = "MDP_Solver/saved_mdps/4_9_pm_SH1/SH1_0_MDP.p"

    # Load all testing contexts
    context_set_path = 'SH1_context_set.json'
    test_context_set = json.load(open(context_set_path, 'rb'))

    # Create a generator from test_context_set
    make_env_parameters = {"observe_lambda": False,
                           "device": "cpu",
                           "terminal_backlog": 5000,
                           "inverse_reward": True,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": False,
                           "convergence_threshold": 0.1,
                           "terminate_on_lta_threshold": False, }

    env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=456)

    base_env = env_generator.sample(env_id)
    env_generator.clear_history()

    input_shape = torch.Tensor([base_env.observation_spec["Q"].shape[0] + base_env.observation_spec["Y"].shape[0]]).int()
    output_shape = base_env.action_spec.space.n

    # Create mdp agent
    mdp = pickle.load(open(os.path.join(PROJECT_DIR, mdp_path), 'rb'))
    mdp_module = MDP_module(mdp)
    mdp_agent = MDP_actor(mdp_module)

    # Create MaxWeight agent
    mw_agent = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    print("Collecting trajectories using agent and maxweight policy")

    # Collect trajectories from the MDP agent and the MaxWeight policy
    # or load them if they have already been collected
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        try:
            tds = pickle.load(open("tds.pkl", 'rb'))
            print("Loaded tds")
        except FileNotFoundError:
            tds = []
            for n in range(num_rollouts):
                env = env_generator.sample(env_id)
                print(f"Collecting trajectory {n} from MDP agent")
                td = env.rollout(policy=mdp_agent, max_steps=rollout_length)
                tds.append(td)
            pickle.dump(tds, open("tds.pkl", 'wb'))
        try:
            mw_tds = pickle.load(open("mw_tds.pkl", 'rb'))
            print("Loaded mw_tds")
        except FileNotFoundError:
            mw_tds = []
            for n in range(num_rollouts):
                env = env_generator.sample(env_id)
                print(f"Collecting trajectory {n} from MaxWeight policy")
                mw_td = env.rollout(policy=mw_agent, max_steps=rollout_length)
                mw_tds.append(mw_td)
            pickle.dump(mw_tds, open("mw_tds.pkl", 'wb'))

    # Compute the mean lta for the mdp agent and mw agent's trajectories
    mdp_ltas = [compute_lta(td["backlog"]) for td in tds]
    mdp_mean_lta = np.mean(mdp_ltas, axis=0)
    mdp_std_lta = np.std(mdp_ltas, axis=0)

    mw_ltas = [compute_lta(mw_td["backlog"]) for mw_td in mw_tds]
    mw_mean_lta = np.mean(mw_ltas, axis=0)
    mw_std_lta = np.std(mw_ltas, axis=0)


    # plot the lta of the mdp agent and the maxweight policy
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(mdp_mean_lta, label="MDP Agent")
    ax.fill_between(range(len(mdp_mean_lta)), mdp_mean_lta - mdp_std_lta, mdp_mean_lta + mdp_std_lta, alpha=0.5)
    ax.plot(mw_mean_lta, label="MaxWeight")
    ax.fill_between(range(len(mw_mean_lta)), mw_mean_lta - mw_std_lta, mw_mean_lta + mw_std_lta, alpha=0.5)
    ax.set_ylim(0, mw_mean_lta.max() * 2)
    ax.legend()
    ax.set_title("LTA Comparison between MDP Agent and MaxWeight Policy")
    fig.show()

    # Create the training dataset by:
    #   Merging all the mlp_trajectories in tds into a single tensor dict
    #   modifying the td to have a "true_action" key
    # Merge all tensordicts in tds into a single tensordict
    td = torch.cat(tds)
    td["true_action"] = td["action"].clone()

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size = td.shape[0]),
                                 batch_size = int(td.shape[0]/100),
                                 sampler = SamplerWithoutReplacement())
    replay_buffer.extend(td)
    sample = replay_buffer.sample()

    # Create an MLP agent
    MLP_agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["Q", "Y"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        actor_depth=actor_depth,
        actor_cells=actor_cells,
    )

    policy_mlp = MLP_agent.get_policy_operator()

    # Train the MLP agent
    train_module2(policy_mlp, replay_buffer, num_training_epochs=training_epochs, lr=0.0001, loss_fn=nn.BCELoss())
    # generator a trajectory from the MLP agent agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        mlp_nn_tds = []
        for n in range(num_rollouts):
            print(f"Collecting trajectory {n} from MLP agent")
            env = env_generator.sample(env_id)
            mlp_nn_td = env.rollout(policy=MLP_agent, max_steps=rollout_length)
            mlp_nn_tds.append(mlp_nn_td)

    # compute the mean lta for the mlp agent's trajectories
    mlp_nn_ltas = [compute_lta(mlp_nn_td["backlog"]) for mlp_nn_td in mlp_nn_tds]
    mlp_nn_mean_lta = np.mean(mlp_nn_ltas, axis=0)
    mlp_nn_std_lta = np.std(mlp_nn_ltas, axis=0)


    # Plot the LTA for all agents
    arrival_rates = env.base_env.arrival_rates
    # mw_nn_lta = compute_lta(mw_nn_td["backlog"])
    mlp_nn_lta = compute_lta(mlp_nn_td["backlog"])
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # plot the arrival rates as a bar chart
    ax[0].bar(range(1, len(arrival_rates) + 1), arrival_rates)
    ax[0].set_title("Arrival Rates")
    ax[0].set_xlabel("Node")
    # Plot LTA for all trajectories
    ax[1].plot(mdp_mean_lta, label="MDP Agent")
    ax[1].fill_between(range(len(mdp_mean_lta)), mdp_mean_lta - mdp_std_lta, mdp_mean_lta + mdp_std_lta, alpha=0.2)
    ax[1].plot(mw_mean_lta, label="MaxWeight")
    ax[1].fill_between(range(len(mw_mean_lta)), mw_mean_lta - mw_std_lta, mw_mean_lta + mw_std_lta, alpha=0.2)
    ax[1].plot(mlp_nn_mean_lta, label=f"MLP ({actor_depth}, {actor_cells}) NN Agent")
    ax[1].fill_between(range(len(mlp_nn_mean_lta)), mlp_nn_mean_lta - mlp_nn_std_lta, mlp_nn_mean_lta + mlp_nn_std_lta, alpha=0.2)
    ax[1].legend()
    ax[1].set_title("Agent performance Comparison")
    arrival_rates_formatted = [float(f"{x:.2f}") for x in arrival_rates]
    # mw_nn_error_rate_formatted = f"{mw_nn_error_rate:.4f}"
    # w_normalized = [float(f"{x:.4f}") for x in w_normalized]

    fig.suptitle(f"Imitation Learning LTA Comparison {env_id}")
    fig.show()

