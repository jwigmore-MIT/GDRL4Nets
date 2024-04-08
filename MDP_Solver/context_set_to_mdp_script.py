from torchrl_development.envs.env_generators import make_env, parse_env_json
import numpy as np
from SingleHopMDP import SingleHopMDP
import json
import os

max_iterations = 200

context_set_dict = json.load(open("SH1_context_set.json", 'rb'))
for i in range(context_set_dict["num_envs"]):
    context_dict = context_set_dict["context_dicts"][str(i)]
    env_params = context_dict["env_params"]
    env = make_env(env_params)
    mdp = SingleHopMDP(env, name = f"SH1_{i}_MDP", q_max = np.array(context_dict["Q_max"])*1.5)
    mdp.compute_tx_matrix(save_path = "saved_tx_matrices")
    mdp.do_VI(max_iterations = max_iterations, theta = 0.1)
    mdp.save_MDP(os.path.join('saved_mdps', f"SH1_{i}_MDP.p"))
