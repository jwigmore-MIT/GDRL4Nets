from torchrl_development.envs.env_generators import make_env, parse_env_json
import numpy as np
from SingleHopMDP import SingleHopMDP
import json
import os
import argparse

def smart_type(value):

    if ',' in value:
        try:
            value_list = [float(item) for item in value.split(',')]
            return np.array(value_list)
        except ValueError:
            pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass


    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    return value

parser = argparse.ArgumentParser(description='Run experiment')

parser.add_argument('--max_iterations', type=int, default=100)
parser.add_argument('--q_max', type=int, default=50)

args = parser.parse_args()


max_iterations = args.max_iterations
q_max = args.q_max

context_set_dict = json.load(open("context_sets/SH1_poisson_context_set.json", 'rb'))
# for i in range(context_set_dict["num_envs"]):
for i in [0, 1,2]:
    context_dict = context_set_dict["context_dicts"][str(i)]
    env_params = context_dict["env_params"]
    env = make_env(env_params)
    mdp = SingleHopMDP(env, name = f"SH1_Poisson_{i}_MDP", q_max = q_max)
    mdp.compute_tx_matrix(save_path = "saved_tx_matrices")
    # mdp.load_tx_matrix("saved_tx_matrices/SH1_1_MDP_qmax60_discount0.99_computed_tx_matrix.pkl")
    mdp.do_VI(max_iterations = max_iterations, theta = 0.1)
    # check of the saved_mdps directory exists, if it doesn't create it
    if not os.path.exists("saved_mdps"):
        os.makedirs("saved_mdps")
    save_path = os.path.join("saved_mdps", f"SH1_Poisson_{i}_MDP.p")
    # check if the save_path already exists, if it does add a number to the end of the file name
    if os.path.exists(save_path):
        j = 1
        while os.path.exists(save_path):
            save_path = os.path.join("saved_mdps", f"SH1_Poisson_{i}_MDP_{j}.p")
            j += 1
    mdp.save_MDP(save_path)
