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

base_env_generator = EnvGenerator(poisson_context, make_env_parameters, env_generator_seed=456)
base_env = base_env_generator.sample(0)

input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n

# Next we load the models for each agent
file_dict = {"Poisson": {  # context type
    "MLP": { # agent type
        0: "poisson0_mlp_model_1105000.pt", # context number
        1: "poisson1_mlp_model_2905000.pt",
        2: "poisson2_mlp_model_1605000.pt"
    } ,
    "MWN": {
       0: "poisson0_mwnn_model_1105000.pt",
       1: "poisson1_mwnn_model_2905000.pt",
       2: "poisson2_mwnn_model_1605000.pt"
    }
}}

#
agents = {}
for context_type in file_dict.keys():
    agents[context_type] = {}
    for agent_type in file_dict[context_type].keys():
        agents[context_type][agent_type] = {}
        for context_num in file_dict[context_type][agent_type].keys():
            if agent_type == "MLP":
                agent = create_actor_critic(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    in_keys=["observation"],
                    action_spec=base_env.action_spec,
                    temperature=0.1,
                    actor_depth = 2,
                    actor_cells = 64,
                )
                agent.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "experiments/experiment12/Trained_Agent_comparison/trained_agents", file_dict[context_type][agent_type][context_num])))
                agents[context_type][agent_type][context_num] = agent
            elif agent_type == "MWN":
                agent = create_maxweight_actor_critic(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    in_keys=["Q", "Y"],
                    action_spec=base_env.action_spec,
                    temperature=10,
                )
                agent.load_state_dict(torch.load(os.path.join(PROJECT_DIR, "experiments/experiment12/Trained_Agent_comparison/trained_agents", file_dict[context_type][agent_type][context_num])))
                agents[context_type][agent_type][context_num] = agent




