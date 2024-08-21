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

from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))


"""
Want to asses the performance of the MWN and MLP agents on all three SH3 contexts


"""



rollout_length =30000
num_rollouts = 3
env_generator_seed = 4162024
context_ids = {'a': 0, 'b': 1, 'c': 2}
lta_tds = {}
test_context_set_path = 'SH3_context_set_100_03251626.json'
context_set = json.load(open(test_context_set_path, 'rb'))

# Create environment generator
make_env_parameters = {"observe_lambda": False,
                        "device": 'cpu',
                        "terminal_backlog": None,
                        "inverse_reward": True,
                        "stat_window_size": 100000,
                        "terminate_on_convergence": False,
                        "convergence_threshold": 0.01,
                        "terminate_on_lta_threshold": False,}

env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=env_generator_seed)
base_env = env_generator.sample(0)
input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n


agent_dir = 'SH3_trained_agents'

agent_dict = {}
# iterate through all agents in agent_dir
print("Loading agents...")
for file in os.listdir(agent_dir):
    if file.endswith(".pt"):
        # split the file name to get the agent_type and training environment
        agent_type, env_id = file.split("_")[0], file.split("_")[1]
        # load either MWN agent or MLP agent based on agent_type
        if agent_type == "MWN":
            agent = create_maxweight_actor_critic(
                input_shape,
                output_shape,
                in_keys=["Q", "Y"],
                action_spec=base_env.action_spec,
                temperature=5,
            )
        elif agent_type == "MLP":
            agent = create_actor_critic(
                input_shape,
                output_shape,
                in_keys=["observation"],
                action_spec=base_env.action_spec,
                temperature=0.1,
                actor_depth=2,
                actor_cells=64
            )
        agent.load_state_dict(torch.load(os.path.join(agent_dir, file)))
        # Store agents
        agent_dict[(agent_type, env_id)] = agent


# Test each agent on each environment
lta_tds = {}
means = {}
stds = {}

print("Testing agents...")
pbar = tqdm(total=context_ids.keys().__len__() * len(agent_dict)*num_rollouts)

with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
    for env_char ,env_num in context_ids.items():
        for (agent_type, training_env), agent in agent_dict.items():
            env = env_generator.sample(env_num)
            ltas = []

            for n in range(num_rollouts):
                pbar.set_description(f"Testing agent {agent_type} on environment {env_num}: {n}/{num_rollouts}")

                env.reset()
                td = env.rollout(policy = agent, max_steps = rollout_length)
                ltas.append(compute_lta(td["backlog"]))
                pbar.update(1)
            training_env_num = context_ids[training_env[-1]]
            lta_tds[(env_num, agent_type, training_env_num)] = torch.stack(ltas)
            means[(env_num, agent_type, training_env_num)] = torch.stack(ltas).mean(dim=0)[-1].item()
            stds[(env_num, agent_type, training_env_num)] = torch.stack(ltas).std(dim=0)[-1].item()
            # reset env_generator seed
            env_generator.reseed(env_generator_seed)


""" Plot results
The results keys are (testing_env_id, agent_type, training_env_id)
We want a single bar plot for each testing_env_id with a bar for each agent_type, training_env_id combination
Each bar has the mean lta of the agent on the testing environment with error bars representing the standard deviation
"""


bar_width = 0.35
n_environments = len(context_ids)

for test_env_id in context_ids.values():
    fig, ax = plt.subplots()
    index = np.arange(n_environments)

    means_list_mwn = [means[(test_env_id, "MWN", training_env)] for env_char, training_env in context_ids.items()]
    stds_list_mwn = [stds[(test_env_id, "MWN", training_env)] for env_char, training_env in context_ids.items()]
    ax.bar(index - bar_width / 2, means_list_mwn, bar_width, yerr=stds_list_mwn, label="MWN")

    means_list_mlp = [means[(test_env_id, "MLP", training_env)] for env_char, training_env in context_ids.items()]
    stds_list_mlp = [stds[(test_env_id, "MLP", training_env)] for env_char, training_env in context_ids.items()]
    ax.bar(index + bar_width / 2, means_list_mlp, bar_width, yerr=stds_list_mlp, label="MLP")

    ax.set_xticks(index)
    ax.set_xticklabels(context_ids.values())
    ax.set_xlabel("Training Environment")
    ax.set_ylabel("Mean LTA")
    ax.legend()
    ax.set_title(f"Environment {test_env_id}")
    plt.show()

# plot again but then normalize by the max weight lta
for test_env_id in context_ids.values():
    fig, ax = plt.subplots()
    index = np.arange(n_environments)

    max_weight_lta = context_set["context_dicts"][str(test_env_id)]["lta"]

    means_list_mwn = [means[(test_env_id, "MWN", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
    stds_list_mwn = [stds[(test_env_id, "MWN", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
    ax.bar(index - bar_width / 2, means_list_mwn, bar_width, yerr=stds_list_mwn, label="MWN")

    means_list_mlp = [means[(test_env_id, "MLP", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
    stds_list_mlp = [stds[(test_env_id, "MLP", training_env)]/max_weight_lta for env_char, training_env in context_ids.items()]
    ax.bar(index + bar_width / 2, means_list_mlp, bar_width, yerr=stds_list_mlp, label="MLP")

    ax.set_xticks(index)
    ax.set_xticklabels(context_ids.values())
    ax.set_xlabel("Training Environment")
    ax.set_ylabel("Normalized Mean LTA")

    ax.legend()
    ax.set_title(f"Environment {test_env_id}: Normalized Performance")
    plt.show()


# Now plot normalized performance as a function of time by iterating through the lta_tds dict
for test_env_id in context_ids.values():
    fig, ax = plt.subplots()
    for training_env_id in context_ids.values():
        max_weight_lta = context_set["context_dicts"][str(test_env_id)]["lta"]

        mwn_ltas = lta_tds[(test_env_id, "MWN", training_env_id)]/max_weight_lta
        mlp_ltas = lta_tds[(test_env_id, "MLP", training_env_id)]/max_weight_lta

        mean_mwn = mwn_ltas.mean(dim=0)
        std_mwn = mwn_ltas.std(dim=0)

        mean_mlp = mlp_ltas.mean(dim=0)
        std_mlp = mlp_ltas.std(dim=0)

        ax.plot(mean_mwn, label=f"MWN trained on Environment {training_env_id}")
        ax.fill_between(range(len(mean_mwn)), mean_mwn - std_mwn, mean_mwn + std_mwn, alpha=0.2)

        ax.plot(mean_mlp, label=f"MLP trained on Environment {training_env_id}")
        ax.fill_between(range(len(mean_mlp)), mean_mlp - std_mlp, mean_mlp + std_mlp, alpha=0.2)

    ax.set_xlabel("Time")
    ax.set_ylabel("Normalized Performance")
    ax.legend()
    ax.set_title(f"Environment {test_env_id}: Normalized Performance over Time")
    plt.show()


# Get all of the weights from the MWN agents
mwn_weights = {}
normalized_mwn_weights = {}
for (agent_type, env_id), agent in agent_dict.items():
    if agent_type == "MWN":
        state_dict = agent.get_policy_operator().state_dict()
        mwn_weights[env_id] = state_dict['module.0.module.weights'].detach().numpy()
        normalized_mwn_weights[env_id] = mwn_weights[env_id]/mwn_weights[env_id].sum(axis=0, keepdims=True)


# Get all of the arrival rates for each of the contexts
arrival_rates = {}
for env_id in context_ids.values():
    arrival_rates[env_id] = context_set["context_dicts"][str(env_id)]["arrival_rates"]

# Get the service rate for each of the contexts
service_rates = {}
for env_id in context_ids.values():
    context_dict = context_set["context_dicts"][str(env_id)]
    service_rates[env_id] = [params["service_rate"] for class_id,params in context_dict["env_params"]["Y_params"].items()]
# plot the normalized MWN weights as a bar plot for each environment,a and plot the arrival rates
# in the secod column
fig, axes = plt.subplots(nrows=len(normalized_mwn_weights), ncols=2, figsize=(10, 10))
ax_row = -1
for env_id, weights in normalized_mwn_weights.items():
    ax_row +=1
    weights_ax = axes[ax_row][0]
    arrival_rates_ax = axes[ax_row][1]
    index = np.arange(1,weights.shape[0]+1)
    weights_ax.bar(index, weights)
    weights_ax.set_title(f"Environment {ax_row+1}: Normalized MWN Weights")
    weights_ax.set_ylim(0, 0.5)

    arrival_rate = arrival_rates[ax_row]
    service_rate = service_rates[ax_row]
    normalized_arrival_rates = arrival_rate/np.array(service_rate)
    arrival_rates_ax.bar(index, normalized_arrival_rates)
    arrival_rates_ax.set_title(f"Environment {ax_row+1}: Arrival Rate/Service Rate")
    arrival_rates_ax.set_ylim(0, 0.5)
    arrival_rates_ax.set_xlabel("Class")

fig.tight_layout()
plt.show()


# Where do the agents differ?
"""
Where do the agents differ? I 
"""



# for env_id in range(env_generator.num_envs):
#     fig, ax = plt.subplots()
#     for agent_id in mdp_agents.keys():
#         max_weight_lta = test_context_set["context_dicts"][str(env_id)]["lta"]
#         ltas = results[(env_id, agent_id)]
#         mean = ltas.mean(dim=0)/max_weight_lta
#         std = ltas.std(dim=0)/(max_weight_lta**2)
#         ax.plot(mean, label=f"Agent: {agent_id}")
#         ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
#     ax.legend()
#     ax.set_title(f"Environment {env_id}")
#     plt.show()


