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


if __name__ == "__main__":
    """ 
    This script is to test the performance of MaxWeight on all contexts in the
    SH1_poisson_context_set.json file
    """

    context_set_file = "SH1_poisson_context_set.json"
    context_set = json.load(open(context_set_file, "r"))

    # Make env parameters
    make_env_parameters = {"observe_lambda": False,
                           "device": "cpu",
                           "terminal_backlog": 5000,
                           "inverse_reward": True,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": False,
                           "convergence_threshold": 0.1,
                           "terminate_on_lta_threshold": False, }

    env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=456)

    base_env = env_generator.sample(0)
    env_generator.clear_history()

    # Create MaxWeight Actor
    mw_actor = MaxWeightActor(in_keys= ["Q", "Y"], out_keys= ['action'])

    # Collect three trajectories from each environment in the context set
    num_trajectories = 1
    num_steps = 10000
    tds = {}
    for n in range(env_generator.num_envs):
        tds[n] = []
        for i in range(num_trajectories):
            env = env_generator.sample(n)
            td = env.rollout(policy=mw_actor, max_steps=num_steps)
            tds[n].append(td)

    # Compute the LTA for each environment
    lta_dict = {}
    for n in range(env_generator.num_envs):
        lta_dict[n] = []
        for i in range(num_trajectories):
            lta = compute_lta(tds[n][i]["backlog"])
            lta_dict[n].append(lta)

    # Compute the mean and std for each environment
    mean_lta_dict = {}
    std_lta_dict = {}
    for n in range(env_generator.num_envs):
        mean_lta_dict[n] = np.mean(lta_dict[n], axis=0)
        std_lta_dict[n] = np.std(lta_dict[n], axis=0)

    # Plot the mean LTA for each environment
    fig, ax = plt.subplots()
    for n in range(env_generator.num_envs):
        ax.plot(mean_lta_dict[n], label=f"Env {n}")
        ax.fill_between(range(num_steps), mean_lta_dict[n] - std_lta_dict[n], mean_lta_dict[n] + std_lta_dict[n], alpha=0.3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("LTA")
    ax.legend()
    fig.suptitle("LTA for each environment in SH1_poisson_context_set.json")
    plt.show()

    # Repeat for the SH1_discrete_context_set.json

    context_set_file = "SH1_discrete_context_set.json"
    context_set = json.load(open(context_set_file, "r"))

    env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=456)

    # storage for the discrete context set
    d_tds = {}
    d_ltas = {}
    d_mean_ltas = {}
    d_std_ltas = {}

    for n in range(env_generator.num_envs):
        d_tds[n] = []
        d_ltas[n] = []
        for i in range(num_trajectories):
            env = env_generator.sample(n)
            td = env.rollout(policy=mw_actor, max_steps=num_steps)
            d_tds[n].append(td)
            lta = compute_lta(td["backlog"])
            d_ltas[n].append(lta)
        d_mean_ltas[n] = np.mean(d_ltas[n], axis=0)
        d_std_ltas[n] = np.std(d_ltas[n], axis=0)

    fig, ax = plt.subplots()
    for n in range(env_generator.num_envs):
        ax.plot(d_mean_ltas[n], label=f"Env {n}")
        ax.fill_between(range(num_steps), d_mean_ltas[n] - d_std_ltas[n], d_mean_ltas[n] + d_std_ltas[n], alpha=0.3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("LTA")
    ax.legend()
    fig.suptitle("LTA for each environment in SH1_discrete_context_set.json")
    plt.show()

    # Measure the average standard deviation of LTA for each environment
    # in both context sets
    mean_std_lta = {}
    d_mean_std_ltas = {}
    for n in range(env_generator.num_envs):

        mean_std_lta[n] = np.round(np.mean(std_lta_dict[n]), 3)
        d_mean_std_ltas[n] = np.round(np.mean(d_std_ltas[n]), 3)

    print(f"Mean standard deviation of LTA for each environment in SH1_poisson_context_set.json: {mean_std_lta}")
    print(f"Mean standard deviation of LTA for each environment in SH1_discrete_context_set.json: {d_mean_std_ltas}")

    # Measure the mean and variance in each Q_state
    q_vars = {}
    q_means = {}
    d_q_vars = {}
    d_q_means = {}
    for n in range(env_generator.num_envs):
        q_vars[n] = []
        q_means[n] = []
        d_q_vars[n] = []
        d_q_means[n] = []
        for i in range(num_trajectories):
            q_vars[n].append(tds[n][i]["Q"].var(dim = 0))
            d_q_vars[n].append(d_tds[n][i]["Q"].var(dim = 0))
            q_means[n].append(tds[n][i]["Q"].mean(dim = 0))
            d_q_means[n].append(d_tds[n][i]["Q"].mean(dim = 0))
        q_vars[n] = torch.stack(q_vars[n], dim=0).mean(dim = 0)
        d_q_vars[n] = torch.stack(d_q_vars[n], dim=0).mean(dim = 0)
        q_means[n] = torch.stack(q_means[n], dim=0).mean(dim = 0)
        d_q_means[n] = torch.stack(d_q_means[n], dim=0).mean(dim = 0)

    # Measure the mean and variance in each Y_state
    y_vars = {}
    y_means = {}
    d_y_vars = {}
    d_y_means = {}
    for n in range(env_generator.num_envs):
        y_vars[n] = []
        y_means[n] = []
        d_y_vars[n] = []
        d_y_means[n] = []
        for i in range(num_trajectories):
            y_vars[n].append(tds[n][i]["Y"].var(dim = 0))
            y_means[n].append(tds[n][i]["Y"].mean(dim = 0))
            d_y_vars[n].append(d_tds[n][i]["Y"].var(dim = 0))
            d_y_means[n].append(d_tds[n][i]["Y"].mean(dim = 0))
        y_vars[n] = torch.stack(y_vars[n], dim=0).mean(dim = 0)
        y_means[n] = torch.stack(y_means[n], dim=0).mean(dim = 0)
        d_y_vars[n] = torch.stack(d_y_vars[n], dim=0).mean(dim = 0)
        d_y_means[n] = torch.stack(d_y_means[n], dim=0).mean(dim = 0)

    # Now lets measure the Mask frequency
    """
    In td["mask"] there are four possible values:
    [1, 0, 0]
    [0, 1, 0]
    [0, 0, 1]
    [0, 1, 1]
    where 0 is False and 1 is True
    
    I want to measure the frequency of each of these masks for environment 0
    for each context
    """

    mask_freq = {}
    d_mask_freq = {}
    for n in range(env_generator.num_envs):
        mask_freq[n] = {}
        d_mask_freq[n] = {}
        for i in range(num_trajectories):
            mask_freq[n][i] = {}
            d_mask_freq[n][i] = {}
            for j in range(num_steps):
                mask = tds[n][i]["mask"][j]
                d_mask = d_tds[n][i]["mask"][j]
                mask = tuple(mask.tolist())
                d_mask = tuple(d_mask.tolist())
                if mask not in mask_freq[n][i]:
                    mask_freq[n][i][mask] = 1
                else:
                    mask_freq[n][i][mask] += 1
                if d_mask not in d_mask_freq[n][i]:
                    d_mask_freq[n][i][d_mask] = 1
                else:
                    d_mask_freq[n][i][d_mask] += 1

    # Now compare
    # Measure the arrival rate for each context
    arrival_rates = {}
    d_arrival_rates = {}
    for n in range(env_generator.num_envs):
        arrival_rates[n] = []
        d_arrival_rates[n] = []
        for i in range(num_trajectories):
            arrival_rates[n].append(tds[n][i]["arrivals"].mean(dim = 0))
            d_arrival_rates[n].append(d_tds[n][i]["arrivals"].mean(dim = 0))
