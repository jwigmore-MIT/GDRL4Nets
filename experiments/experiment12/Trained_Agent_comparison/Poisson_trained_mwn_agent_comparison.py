import json
import os
from torchrl_development.utils.configuration import load_config
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic
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

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))


"""
Here we compare the perforformance of agents trained on a single Poisson context to agents trained
on a single Discrete context. The Poisson and Discrete contexts have the same average service and arrival rates

We also have two types of agents: MLP and MW_NN

There are three contexts for each type of context. There is a single agent of each time trained on each context, meaning we have 
a total of 12 agents to compare.

We want to compare performance on:
1. The training context
2. The other contexts of the same type
3. The equivalent context of the other type
4. The other contexts of the other type
"""

# Hyperparameters
env_generator_seed = 531997
num_rollouts = 2
max_steps = 30000



# First we load the context sets
poisson_context = json.load(open(os.path.join(PROJECT_DIR, "experiments/experiment12/SH1_poisson_context_set.json"), "r"))
discrete_context = json.load(open(os.path.join(PROJECT_DIR, "experiments/experiment12/SH1_discrete_context_set.json"), "r"))

# Create a base env_generator
make_env_parameters = {"observe_lambda": False,
                          "device": "cpu",
                            "terminal_backlog": 5000,
                            "inverse_reward": True,
                            "stat_window_size": 100000,
                            "terminate_on_convergence": False,
                            "convergence_threshold": 0.1,
                            "terminate_on_lta_threshold": False, }

env_generator = EnvGenerator(poisson_context, make_env_parameters, env_generator_seed=env_generator_seed)
base_env = env_generator.sample(0)

input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n

# Next we load the models for each agent
file_dict = {"Poisson": {  # context type
    "MWN": {
       0: "poisson_a_mwn_model_2905000.pt",
       1: "poisson_b_mwn_model_2905000.pt",
       2: "poisson_c_mwn_model_2905000.pt"
    },
    "MLP": {
        0: "poisson_a_mlp_model_2905000.pt",
        1: "poisson_b_mlp_model_2905000.pt",
        2: "poisson_c_mlp_model_2905000.pt"

    }
}}

#
agents = {}
for context_type in file_dict.keys():
    agents[context_type] = {}
    for agent_type in file_dict[context_type].keys():
        agents[context_type][agent_type] = {}
        for context_num in file_dict[context_type][agent_type].keys():
            if agent_type == "MWN":
                agent = create_maxweight_actor_critic(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    in_keys=["Q", "Y"],
                    action_spec=base_env.action_spec,
                    temperature=10,
                )
                agent.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "experiments/experiment12/Trained_Agent_comparison/trained_agents", file_dict[context_type][agent_type][context_num])))
                agents[context_type][agent_type][f"MWN_{context_num}"] = agent
            elif agent_type == "MLP":
                agent = create_actor_critic(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    in_keys=["observation"],
                    action_spec=base_env.action_spec,
                    temperature=0.1,
                    actor_depth=2,
                    actor_cells=64,
                )
                agent.load_state_dict(torch.load(
                    os.path.join(PROJECT_DIR, "experiments/experiment12/Trained_Agent_comparison/trained_agents",
                                 file_dict[context_type][agent_type][context_num])))
                agents[context_type][agent_type][f"MLP_{context_num}"] = agent
#



# Test each agent on all contexts num_rollouts times
results = {}
with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
    for context_type in agents.keys(): # Poisson or Discrete
        results[context_type] = {}
        for agent_type in agents[context_type].keys(): # MWN or MLP
            results[context_type][agent_type] = {}
            for training_context in agents[context_type][agent_type].keys(): # 0, 1, 2
                env_generator.reseed(env_generator_seed)
                results[context_type][agent_type][training_context] = {}
                agent = agents[context_type][agent_type][training_context] # get agent trained on this context
                for testing_context in range(env_generator.num_envs): #
                    testing_context_str = f"context_{testing_context}"
                    results[context_type][agent_type][training_context][testing_context_str] = {}
                    for n in range(num_rollouts):
                        env = env_generator.sample(testing_context)
                        td = env.rollout(max_steps = max_steps, policy=agent)
                        results[context_type][agent_type][training_context][testing_context_str][n] = compute_lta(td["backlog"]).unsqueeze(1)
                # take average over all contexts
                    results[context_type][agent_type][training_context][testing_context_str]["mean"] = torch.cat([results[context_type][agent_type][training_context][testing_context_str][n] for n in range(num_rollouts)], dim = 1).mean(dim =1)
                    results[context_type][agent_type][training_context][testing_context_str]["std"] = torch.cat([results[context_type][agent_type][training_context][testing_context_str][n] for n in range(num_rollouts)], dim =1).std(dim =1)


# plot the result of the poisson context set, MWN agents,
if False:
    mwn_results = results["Poisson"]["MWN"]
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharey = False, sharex = True)

    for num ,agent_num in enumerate(mwn_results.keys()):
        ax = axs[num]
        ax.set_title(f"Agent {agent_num}")
        for context_num in mwn_results[agent_num].keys():
            mean = mwn_results[agent_num][context_num]["mean"]
            std = mwn_results[agent_num][context_num]["std"]
            ax.plot(mean, label=context_num)
            ax.fill_between(range(len(mean)), mean-std, mean+std, alpha = 0.2)
        ax.legend()

    fig.tight_layout()
    plt.show()

if False:
# redo the plots but this time normalize the mean by the lta in the context_set
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharey = False, sharex = True)

    for num ,agent_num in enumerate(mwn_results.keys()):
        ax = axs[num]
        ax.set_title(f"Agent {agent_num}: Normalized Performance")
        for context_num in mwn_results[agent_num].keys():
            mean = mwn_results[agent_num][context_num]["mean"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]
            std = mwn_results[agent_num][context_num]["std"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]**2
            ax.plot(mean, label=context_num)
            ax.fill_between(range(len(mean)), (mean-std), (mean+std), alpha = 0.2)
            ax.set_ylim(0, 1.1)
        ax.legend()

    fig.tight_layout()
    plt.show()


if False:
    # Now make a single plot for each testing context, and plot the performance of each agent on that context
    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharey = False, sharex = True)

    for num, context_num in enumerate(mwn_results["agent_0"].keys()):
        ax = axs[num]
        ax.set_title(f"Context {context_num}")
        for agent_num in mwn_results.keys():
            mean = mwn_results[agent_num][context_num]["mean"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]
            std = mwn_results[agent_num][context_num]["std"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]**2
            ax.plot(mean, label=agent_num)
            ax.fill_between(range(len(mean)), mean-std, mean+std, alpha = 0.2)
        ax.legend()
        # Set the y_lim to be 50% larger than the smallest final mean
        ax.set_ylim(0, 1.5)

    fig.tight_layout()
    plt.show()


# Plot the performance of both the MWN and MLP agents on each context
# Give each context a graph, and plot all agents on the same graph
# Normalize the performance by the LTA of the context
fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharey = False, sharex = True)
total_contexts= 3

for num in range(total_contexts): # this iterates through the evaluation contexts in the poisson context set
    ax = axs[num]
    context_num = f"context_{num}"
    ax.set_title(f"Context {context_num}")
    for agent_type in results["Poisson"].keys(): # this iterates through the agent type (i.e. MWN or MLP)
        y_max = 0.5
        for agent_num in results["Poisson"][agent_type].keys(): # this iterate through the agent number (i.e. training context)
            mean = results["Poisson"][agent_type][agent_num][context_num]["mean"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]
            std = results["Poisson"][agent_type][agent_num][context_num]["std"]/poisson_context['context_dicts'][context_num.split("_")[-1]]["lta"]**2
            if mean.max() > y_max and mean.max() < 2:
                y_max = mean.max()

            ax.plot(mean, label=agent_num)
            ax.fill_between(range(len(mean)), mean-std, mean+std, alpha = 0.2)
    ax.legend()
    ax.set_ylim(0, y_max*1.1)


fig.tight_layout()
plt.show()


