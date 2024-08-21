from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from torchrl_development.actors import MDP_module, MDP_actor
import torch
import numpy as np
import argparse
import os
import pandas as pd
from analysis_functions import *
# os.chdir(os.path.realpath("C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\experiments\\experiment14"))

# set the working directory to the parent directory using only the relative path
#os.chdir(os.path.join(os.getcwd()))


# For running from the command line


"""
The goal of this file is to create a network instance with a small effective state-action space under the MaxWeight and optimal policy.
It is a manual script that will import the topology from the SH1B.json environment file, and allow me to change the 
arrival rates and service rates of the network.  The SH1B network is a 2 queue network with Bernoulli arrivals of rate
$\lambda_i$ and Bernoulli capacities with rate $\mu_i$. 
"""
#%%
# Settings

# MDP Environment
context = "a"



# Configure Base Params
local_path = "Singlehop_Two_Node_Simple_"
full_path = os.path.join(os.getcwd(), local_path+ context + ".json")
base_env_params = parse_env_json(full_path=full_path)
mdp_name = f"SH_{context}"
#


# Make Environment
env = make_env(base_env_params, terminal_backlog=100)

# Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
#%% Now create an MDP from the generated environment
mdp = SingleHopMDP(env, name = mdp_name, q_max = 30, value_iterator = 'minus')
#%% Load MDP
mdp.load_VI(f"saved_mdps/{mdp_name}_qmax40_discount0.99_VI_dict.p")

#%% Create a dataframe from the vi_policy
"""
mdp.vi_policy is a dictionary with keys as states in the format of (q1, q2, y1, y2) and values as the action to take 
in the form as [Bool, Bool, Bool] where only one is true
We want to create a dataframe with columns Q1, Q2, Y1, Y2, Action
"""

vi_policy_df = pd.DataFrame(mdp.vi_policy.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
vi_policy_df["Action"] = list(mdp.vi_policy.values())

"Now convert the action to a integer 0,1,2"
vi_policy_df["Action"] = vi_policy_df["Action"].apply(lambda x: np.argmax(x))


"Now create a df with only the states where Y1=1 and Y2=1"
vi_policy_df = vi_policy_df[(vi_policy_df["Y1"] == 1) & (vi_policy_df["Y2"] == 1)]

"Now we want to create a scatter plot of the decision regions"
axis_keys = ["Q1", "Q2"]
fig, ax = plt.subplots(1,1, figsize = (10,10))
sc, ax = plot_state_action_map(vi_policy_df, [("Y1", 1), ("Y2", 1)], ax = ax, axis_keys = axis_keys, policy_type = "VI", plot_type = "Action", collected_frames = None)

plt.show()


"Get MaxWeights state action map"
mw_df = create_state_action_map_from_model(mw_actor, env, temp = 1, compute_action_prob = False, Y_max=1)

"Repeat the process for the MaxWeight policy"
fig, ax = plt.subplots(1,1, figsize = (10,10))
sc, ax = plot_state_action_map(mw_df, [("Y1", 1), ("Y2", 1)], ax = ax, axis_keys = axis_keys, policy_type = "MW", plot_type = "Action", collected_frames = None)
plt.show()
