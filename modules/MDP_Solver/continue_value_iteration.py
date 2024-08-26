from torchrl_development.envs.env_generators import make_env, parse_env_json
import numpy as np
from SingleHopMDP import SingleHopMDP
import json
import os
import pickle

max_iterations = 100

context_set_dict = json.load(open("SH1_context_set.json", 'rb'))
env_id = 0
context_dict = context_set_dict["context_dicts"][str(env_id)]
env_params = context_dict["env_params"]
env = make_env(env_params)
mdp = pickle.load(open("saved_mdps/4_9_am_SH1/SH1_0_MDP.p", 'rb'))
mdp.env = env
mdp.do_VI(max_iterations = max_iterations, theta = 0.1)
save_path = os.path.join("saved_mdps/4_9_pm_SH1", f"SH1_{env_id}_MDP.p")

# check if the save_path already exists, if it does add a number to the end of the file name
if os.path.exists(save_path):
    j = 1
    while os.path.exists(save_path):
        save_path = os.path.join("saved_mdps", f"SH1_{env_id}_MDP_{j}.p")
        j += 1
mdp.save_MDP(save_path)
