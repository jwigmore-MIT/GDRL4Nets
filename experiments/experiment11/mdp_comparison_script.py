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

    env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=3134)

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
    old_tds = pickle.load(open("tds_old_mdp.pkl", 'rb'))
    mw_tds = pickle.load(open("mw_tds.pkl", 'rb'))

    # Collect Trajectories from new MDP agent
    tds = []
    for n in range(num_rollouts):
        print(f"Collecting trajectory {n} from MDP agent")
        env = env_generator.sample(env_id)
        td = env.rollout(policy=mdp_agent, max_steps=rollout_length)
        tds.append(td)

    #
    # Compute the mean lta for the mdp agent and mw agent's trajectories
    mdp_ltas = [compute_lta(td["backlog"]) for td in tds]
    mdp_mean_lta = np.mean(mdp_ltas, axis=0)
    mdp_std_lta = np.std(mdp_ltas, axis=0)

    old_mdp_ltas = [compute_lta(td["backlog"]) for td in old_tds]
    old_mdp_mean_lta = np.mean(old_mdp_ltas, axis=0)
    old_mdp_std_lta = np.std(old_mdp_ltas, axis=0)

    mw_ltas = [compute_lta(mw_td["backlog"]) for mw_td in mw_tds]
    mw_mean_lta = np.mean(mw_ltas, axis=0)
    mw_std_lta = np.std(mw_ltas, axis=0)


    # plot the lta of the mdp agent and the maxweight policy
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(mdp_mean_lta, label="New MDP Agent")
    ax.fill_between(range(len(mdp_mean_lta)), mdp_mean_lta - mdp_std_lta, mdp_mean_lta + mdp_std_lta, alpha=0.2)
    ax.plot(old_mdp_mean_lta, label="Old MDP Agent")
    ax.fill_between(range(len(old_mdp_mean_lta)), old_mdp_mean_lta - old_mdp_std_lta, old_mdp_mean_lta + old_mdp_std_lta, alpha=0.5)
    ax.plot(mw_mean_lta, label="MaxWeight")
    ax.fill_between(range(len(mw_mean_lta)), mw_mean_lta - mw_std_lta, mw_mean_lta + mw_std_lta, alpha=0.2)
    ax.set_ylim(0, mw_mean_lta.max() * 2)
    ax.legend()
    ax.set_title("LTA Comparison between MDP Agent and MaxWeight Policy")
    fig.show()

