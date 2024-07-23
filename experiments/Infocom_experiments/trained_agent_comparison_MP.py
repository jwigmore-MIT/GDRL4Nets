import torch
import pickle
import os
import sys
from torchrl_development.mdp_actors import MDP_actor, MDP_module
import json
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
import numpy as np
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_independent_actor_critic
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Test Agents')
# add argument for context set folder
parser.add_argument('--training_set_folder', type=str, help='folder containing agents trained according to a particular training set', default="MP4_0-5")
# add argument for context set file name
parser.add_argument('--context_set_file_name', type=str, help='file name of context set', default="MP4_context_set_l3_m1_s100.json")
# add argument for agent types (list of strings)
parser.add_argument('--agent_types', nargs='+', type=str, help='list of agent types', default=["MLP"])

args = parser.parse_args()


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


"""
Want to asses the performance of the MWN and MLP agents on all three SH3 contexts


"""



training_set_folder = args.training_set_folder
context_set_file_name = args.context_set_file_name
agent_types = args.agent_types



rollout_length = 30000
num_rollouts = 3
env_generator_seed = 4162024
lta_tds = {}
# test_context_set_path = 'SH2u2_context_set_20_07091947.json'
#test_context_set_path = "SH3_context_set_100_03251626.json"
trained_agent_folder = os.path.join("trained_agents",training_set_folder)
test_context_set_path = os.path.join("context_sets", context_set_file_name)
context_set = json.load(open(test_context_set_path, 'rb'))
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


pmn_cfg = load_config(os.path.join(SCRIPT_PATH, 'PMN_Shared_PPO_MP_settings.yaml'))
mlp_cfg = load_config(os.path.join(SCRIPT_PATH, 'MLP_PPO_MP_settings.yaml'))



make_env_parameters = {"graph": False,
                                    "observe_lambda": False,
                                    "observe_mu": True,
                                    "terminal_backlog": None,
                                    "observation_keys": ["Q", "Y"],
                                    "observation_keys_scale": None,
                                    "negative_keys": ["Q"],
                                    "symlog_obs": True,
                                    "symlog_reward": False,
                                    "inverse_reward": True,
                                    "cost_based": False,
                                    "reward_scale": 1.0,
                                    "stat_window_size": 5000}

env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=env_generator_seed)
base_env = env_generator.sample(0)
input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n
action_spec = base_env.action_spec
N = int(base_env.base_env.N)
D = int(input_shape[0]/N)



agent_dict = {}
# iterate through all agents in agent types, and get the agents from the trained_agent_folder
for agent_type in agent_types:
    # get directory in trained_agent_folder that will share the name with agent_type
    agent_dir = os.path.join(trained_agent_folder, agent_type)
    # get the .pt file in the agent_dir, this is the agent's weights
    agent_file = None
    for file in os.listdir(agent_dir):
        if file.endswith(".pt"):
            agent_file = file
            break
    if agent_file is None:
        continue
    # load the agent
    if agent_type == "MWN":
        agent = create_maxweight_actor_critic(
            input_shape,
            in_keys=["Q", "Y", "lambda", "mu"],
            action_spec=base_env.action_spec,
            temperature=5,
            init_weights=torch.ones([1, N])
        )
    elif agent_type == "MLP":
        agent = create_actor_critic(
            input_shape,
            output_shape,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=mlp_cfg.agent.temperature,
            actor_depth=mlp_cfg.agent.hidden_sizes.__len__(),
            actor_cells=mlp_cfg.agent.hidden_sizes[-1],
        )
    elif agent_type == "PMN":
        agent = create_independent_actor_critic(number_nodes=N,
                                                actor_input_dimension=D,
                                                actor_in_keys=["Q", "Y", "lambda", "mu"],
                                                critic_in_keys=["observation"],
                                                action_spec=action_spec,
                                                temperature=pmn_cfg.agent.temperature,
                                                actor_depth=pmn_cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=pmn_cfg.agent.hidden_sizes[-1],
                                                type=3,
                                                network_type="PMN",
                                                relu_max=getattr(pmn_cfg, "relu_max", 10),
                                                add_zero = False
                                                )

    agent.load_state_dict(torch.load(os.path.join(agent_dir, agent_file)))
    # Store agents
    agent_dict[agent_type] = agent


# Test each agent on each environment, and store the results in the dictionary
# The results will contain the mean and standard deviation of the lta of the agent on the environment
results = {key: {} for key in agent_dict.keys()}

context_ids = list(env_generator.context_dicts.keys())
print("Testing agents...")
pbar = tqdm(total= len(agent_dict)*num_rollouts*len(context_ids))

with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
    for env_num in context_ids:
        for agent_type, agent in agent_dict.items():
            actor = agent.get_policy_operator()
            actor.eval()
            env = env_generator.sample(env_num)
            ltas = []
            results[agent_type][env_num] = {}
            for n in range(num_rollouts):
                pbar.set_description(f"Testing agent {agent_type} on environment {env_num}: {n}/{num_rollouts}")

                env.reset()
                td = env.rollout(policy = actor, max_steps = rollout_length)
                ltas.append(compute_lta(td["backlog"]))
                pbar.update(1)
            # training_env_num = context_ids[training_env[-1]]
            lta_tds[(env_num, agent_type, env_num)] = torch.stack(ltas)
            # store the means and std of the ltas in the results dictionary
            results[agent_type][env_num]["mean"] = torch.stack(ltas).mean(dim=0)[-1].item()
            results[agent_type][env_num]["std"] = torch.stack(ltas).std(dim=0)[-1].item()

            # reset env_generator seed
            env_generator.reseed(env_generator_seed)

# now pickle each of the agents results in their respective directories
for agent_type, agent_results in results.items():
    with open(f"{trained_agent_folder}/{agent_type}/{agent_type}_results.pkl", "wb") as f:
        pickle.dump(agent_results, f)


# results = {"lta_tds": lta_tds, "means": means, "stds": stds}
# with open(f"{agent_dir}/{agent_dir}_results.pkl", "wb") as f:
#     pickle.dump(results, f)
""" Plot results
The results keys are (testing_env_id, agent_type, training_env_id)
We want a single bar plot for each testing_env_id with a bar for each agent_type, training_env_id combination
Each bar has the mean lta of the agent on the testing environment with error bars representing the standard deviation
"""


# plot the results by iterating through the results dictionary






#
# bar_width = 0.35
# n_environments = len(context_ids)
#
#
# fig, ax = plt.subplots()
# index = np.arange(n_environments)
#
# means_list_pmn = [means[(training_env, "PMN", training_env)] for training_env in context_ids]
# stds_list_pmn = [stds[(training_env, "PMN", training_env)] for training_env in context_ids]
# ax.bar(index - bar_width / 2, means_list_pmn, bar_width, yerr=stds_list_pmn, label="PMN")
#
# means_list_mlp = [means[(training_env, "MLP", training_env)] for training_env in context_ids]
# stds_list_mlp = [stds[(training_env, "MLP", training_env)] for training_env in context_ids]
# ax.bar(index + bar_width / 2, means_list_mlp, bar_width, yerr=stds_list_mlp, label="MLP")
#
# ax.set_xticks(index)
# ax.set_xticklabels(context_ids)
# ax.set_xlabel("Training Environment")
# ax.set_ylabel("Mean LTA")
# ax.set_ylim(0,200)
# ax.legend()
# plt.show()

# results = {"lta_tds": lta_tds, "means": means, "stds": stds}
# with open(f"{agent_dir}/{agent_dir}_results.pkl", "wb") as f:
#     pickle.dump(results, f)
# #
# # plot again but then normalize by the max weight lta
# for test_env_id in context_ids.values():
#     fig, ax = plt.subplots()
#     index = np.arange(n_environments)
#
#     max_weight_lta = context_set["context_dicts"][str(test_env_id)]["lta"]
#
#     means_list_pmn = [means[(test_env_id, "PMN", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
#     stds_list_pmn = [stds[(test_env_id, "PMN", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
#     ax.bar(index - bar_width / 2, means_list_pmn, bar_width, yerr=stds_list_pmn, label="PMN")
#
#     means_list_mlp = [means[(test_env_id, "MLP", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
#     stds_list_mlp = [stds[(test_env_id, "MLP", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
#     ax.bar(index + bar_width / 2, means_list_mlp, bar_width, yerr=stds_list_mlp, label="MLP")
#
#     ax.set_xticks(index)
#     ax.set_xticklabels(context_ids.values())
#     ax.set_xlabel("Training Environment")
#     ax.set_ylabel("Normalized Mean LTA")
#
#     ax.legend()
#     ax.set_title(f"Environment {test_env_id}: Normalized Performance")
#     plt.show()
# #
# #
# # Now plot normalized performance as a function of time by iterating through the lta_tds dict
# for test_env_id in context_ids.values():
#     fig, ax = plt.subplots()
#     for training_env_id in context_ids.values():
#         max_weight_lta = context_set["context_dicts"][str(test_env_id)]["lta"]
#
#         mwn_ltas = lta_tds[(test_env_id, "MWN", training_env_id)]/max_weight_lta
#         mlp_ltas = lta_tds[(test_env_id, "MLP", training_env_id)]/max_weight_lta
#
#         mean_mwn = mwn_ltas.mean(dim=0)
#         std_mwn = mwn_ltas.std(dim=0)
#
#         mean_mlp = mlp_ltas.mean(dim=0)
#         std_mlp = mlp_ltas.std(dim=0)
#
#         ax.plot(mean_mwn, label=f"MWN trained on Environment {training_env_id}")
#         ax.fill_between(range(len(mean_mwn)), mean_mwn - std_mwn, mean_mwn + std_mwn, alpha=0.2)
#
#         ax.plot(mean_mlp, label=f"MLP trained on Environment {training_env_id}")
#         ax.fill_between(range(len(mean_mlp)), mean_mlp - std_mlp, mean_mlp + std_mlp, alpha=0.2)
#
#     ax.set_xlabel("Time")
#     ax.set_ylabel("Normalized Performance")
#     ax.legend()
#     ax.set_title(f"Environment {test_env_id}: Normalized Performance over Time")
#     plt.show()
#
#
#
#
# # Get all of the weights from the MWN agents
# mwn_weights = {}
# normalized_mwn_weights = {}
# for (agent_type, env_id), agent in agent_dict.items():
#     if agent_type == "MWN":
#         state_dict = agent.get_policy_operator().state_dict()
#         mwn_weights[env_id] = state_dict['module.0.module.weights'].detach().numpy()
#         normalized_mwn_weights[env_id] = mwn_weights[env_id]/mwn_weights[env_id].sum(axis=0, keepdims=True)
#
#
# # Get all of the arrival rates for each of the contexts
# arrival_rates = {}
# for env_id in context_ids.values():
#     arrival_rates[env_id] = context_set["context_dicts"][str(env_id)]["arrival_rates"]
#
# # Get the service rate for each of the contexts
# service_rates = {}
# for env_id in context_ids.values():
#     context_dict = context_set["context_dicts"][str(env_id)]
#     service_rates[env_id] = [params["service_rate"] for class_id,params in context_dict["env_params"]["Y_params"].items()]
# # plot the normalized MWN weights as a bar plot for each environment,a and plot the arrival rates
# # in the secod column
# fig, axes = plt.subplots(nrows=len(normalized_mwn_weights), ncols=2, figsize=(10, 10))
# ax_row = -1
# for env_id, weights in normalized_mwn_weights.items():
#     ax_row +=1
#     weights_ax = axes[ax_row][0]
#     arrival_rates_ax = axes[ax_row][1]
#     index = np.arange(1,weights.shape[0]+1)
#     weights_ax.bar(index, weights)
#     weights_ax.set_title(f"Environment {ax_row+1}: Normalized MWN Weights")
#     weights_ax.set_ylim(0, 0.5)
#
#     arrival_rate = arrival_rates[ax_row]
#     service_rate = service_rates[ax_row]
#     normalized_arrival_rates = arrival_rate/np.array(service_rate)
#     arrival_rates_ax.bar(index, normalized_arrival_rates)
#     arrival_rates_ax.set_title(f"Environment {ax_row+1}: Arrival Rate/Service Rate")
#     arrival_rates_ax.set_ylim(0, 0.5)
#     arrival_rates_ax.set_xlabel("Class")
#
# fig.tight_layout()
# plt.show()
#
