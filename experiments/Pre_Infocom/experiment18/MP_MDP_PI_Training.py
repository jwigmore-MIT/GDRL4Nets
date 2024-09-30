from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
from torchrl_development.shortest_queue import ShortestQueueActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from MDP_Solver.MultipathMDP import MultipathMDP
from torchrl_development.actors import MDP_module, MDP_actor
import torch
import numpy as np
import argparse
import os
from analysis_functions import *
# os.chdir(os.path.realpath("C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\experiments\\experiment14"))

# set the working directory to the parent directory using only the relative path
#os.chdir(os.path.join(os.getcwd()))


# For running from the command line
parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--param_key', type=str, help='key to select the parameters for the environment', default="base")
parser.add_argument('--rollout_length', type=int, help='length of the rollout', default=20000)
parser.add_argument('--q_max', type=int, help='maximum queue length', default=20)
parser.add_argument('--max_vi_iterations', type=int, help='maximum number of value iteration iterations', default=100)
parser.add_argument('--continue_training', type=bool, default=True, help='continue training the MDP')
parser.add_argument('--new_mdp', type=bool, default=False, help='Re-do VI from scratch')


args = parser.parse_args()

"""
The goal of this file is to create a network instance with a small effective state-action space under the MaxWeight and optimal policy.
It is a manual script that will import the topology from the SH1B.json environment file, and allow me to change the 
arrival rates and service rates of the network.  The SH1B network is a 2 queue network with Bernoulli arrivals of rate
$\lambda_i$ and Bernoulli capacities with rate $\mu_i$. 
"""
#%%
# Settings
param_key = args.param_key

rollout_length = args.rollout_length
q_max = args.q_max
max_vi_iterations = args.max_vi_iterations
vi_theta = 2
eval_rollouts = 5
eval_seeds = np.arange(1, eval_rollouts+1)



# Configure Base Params
# local_path = "Multipath_Two_Node_Simple_"
# full_path = os.path.join(os.getcwd(), local_path+ args.param_key + ".json")
full_path = os.path.join(os.getcwd(), "MP1.json")
base_env_params = parse_env_json(full_path=full_path)
mdp_name = f"MP_{param_key}"



# Make Environment
env = make_env(base_env_params, terminal_backlog=100)
K = 2

#%% Now create an MDP from the generated environment
mdp = MultipathMDP(env, name = mdp_name, q_max = q_max, value_iterator = 'minus')

# Create MaxWeight Actor
sq_actor = ShortestQueueActor(in_keys=["Q"], out_keys=["action"])

"""Create a TensorDict from mdp.state_list which is a list of states in the format of (q1, q2, y1, y2)
The TensorDict should have keys ["Q", "Y"]

Then make a policy_dict with keys [q1, ..., qk, y1, ..., yk] and values as the action to take in the form of [Bool, Bool, Bool] where only one is true
"""
td = TensorDict({"Q": torch.tensor(mdp.state_list)[:, :K], "Y": torch.tensor(mdp.state_list)[:, K:]}, batch_size=len(mdp.state_list))
td = sq_actor(td)
policy_dict = {tuple(mdp.state_list[i]): td["action"][i].int().tolist() for i in range(len(mdp.state_list))}








# check if tx_matrix exists


try:
    mdp.load_tx_matrix(f"tx_matrices/{mdp_name}_qmax{q_max}_discount0.99_computed_tx_matrix.pkl")
except:
    mdp.compute_tx_matrix(f"tx_matrices")
try:
    mdp.load_pi_policy(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")
    policy_dict = mdp.get_pi_policy_table()
except:
    print("Starting PI from scratch...")
mdp.do_PI(default_policy= policy_dict,max_iterations=100, theta=vi_theta)
mdp.save_pi_policy(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")

#
# if args.new_mdp:
#     print("Starting PI from scratch...")
#     mdp.do_VI(max_iterations=max_vi_iterations, theta=vi_theta)
#     mdp.save_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")
# else:
#     try:
#         print("Loading VI dict...")
#         mdp.load_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
#         if args.continue_training:
#             print("Continuing VI from loaded VI dict...")
#             mdp.do_VI(max_iterations = max_vi_iterations, theta = vi_theta)
#             mdp.save_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
#     except:
#         print("VI dict not found.  Starting VI from scratch...")
#         mdp.do_VI(max_iterations = max_vi_iterations, theta = vi_theta)
#         mdp.save_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
# # %% Create MDP actor
# mdp_actor = MDP_actor(MDP_module(mdp))
#
#
# # %% evaluate both the MaxWeight and MDP policies over three different rollouts/seeds
# results = {}
# for policy_name, actor in {"MDP": mdp_actor, "MW": mw_actor}.items():
#     policy_results = {}
#     for seed in eval_seeds:
#         env = make_env(base_env_params, seed = seed)
#         td = env.rollout(policy=actor, max_steps = rollout_length)
#         lta = compute_lta(td["backlog"])
#         print(f"Actor: {policy_name}, Seed: {seed}, LTA: {lta[-1]}")
#         policy_results[seed] = {"td": td, "lta": lta}
#     results[policy_name] = policy_results
#
# # %% Plot the results
# for policy_name, policy_results in results.items():
#     all_ltas = torch.stack([torch.tensor(policy_results[seed]["lta"]) for seed in eval_seeds])
#     mean_lta = all_ltas.mean(dim = 0)
#     std_lta = all_ltas.std(dim = 0)
#     results[policy_name]["mean_lta"] = mean_lta
#     results[policy_name]["std_lta"] = std_lta
#
# fig, ax = plt.subplots(1,1)
# for policy_name, policy_results in results.items():
#     mean_lta = policy_results["mean_lta"]
#     std_lta = policy_results["std_lta"]
#     ax.plot(policy_results["mean_lta"], label = f"{policy_name} Policy")
#     ax.fill_between(range(len(mean_lta)), mean_lta - std_lta, mean_lta + std_lta, alpha = 0.1)
# ax.set_xlabel("Time")
# ax.set_ylabel("Backlog")
# ax.legend()
# fig.show()
#













