from torchrl_development.envs.env_generators import make_env, parse_env_json
import numpy as np
from SingleHopMDP import SingleHopMDP
import json
import os

max_iterations = 300

context_set_dict = json.load(open("context_sets/SH1_poisson_context_set.json", 'rb'))
# for i in range(context_set_dict["num_envs"]):
for i in [0]:
    context_dict = context_set_dict["context_dicts"][str(i)]
    env_params = context_dict["env_params"]
    env = make_env(env_params)
    mdp = SingleHopMDP(env, name = f"SH1_Poisson_{i}_MDP", q_max = 40)
    mdp.compute_tx_matrix(save_path = "saved_tx_matrices")
    mdp.load_tx_matrix("saved_tx_matrices/SH1_1_MDP_qmax60_discount0.99_computed_tx_matrix.pkl")
    mdp.do_VI(max_iterations = max_iterations, theta = 0.1)
    save_path = os.path.join("saved_mdps", f"SH1_{i}_MDP.p")
    # check if the save_path already exists, if it does add a number to the end of the file name
    if os.path.exists(save_path):
        j = 1
        while os.path.exists(save_path):
            save_path = os.path.join("saved_mdps", f"SH1_{i}_MDP_{j}.p")
            j += 1
    mdp.save_MDP(save_path)
