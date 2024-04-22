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
from MDP_Solver.SingleHopMDP import SingleHopMDP
from tabulate import tabulate


from tqdm import tqdm

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))


"""
Want to see where the MWN agent and the MLP agents differ in terms of their policies.


"""



rollout_length =1000
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


#Test each agent on each environment
tds = {}

print("Testing agents...")
pbar = tqdm(total=context_ids.keys().__len__() * len(agent_dict)*num_rollouts)

with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
    for env_char ,env_num in context_ids.items():
        for (agent_type, training_env), agent in agent_dict.items():
            if agent_type == "MWN":
                pass
            env = env_generator.sample(env_num)
            agent_td = {}
            for n in range(num_rollouts):
                pbar.set_description(f"Testing agent {agent_type} on environment {env_num}: {n}/{num_rollouts}")

                env.reset()
                td = env.rollout(policy = agent, max_steps = rollout_length)
                if agent_type == "MWN":
                    # check if the td["logits"].argmax(dim =1) is the same as td["action"].argmax(dim =1)
                    assert (td["logits"].argmax(dim =1) == td["action"].argmax(dim =1)).all().item()
                agent_td[n] = td
                # check exploration type of agent
                pbar.update(1)
            training_env_num = context_ids[training_env[-1]]
            tds[(env_num, agent_type, training_env_num)] = agent_td

            # reset env_generator seed
            env_generator.reseed(env_generator_seed)

"""
I want to make a map from states to actions for each agent based on the data in tds
"""

# Create a map from states to actions for each agent
state_action_maps = {}
for (env_num, agent_type, training_env_num), agent_td in tds.items():

    state_action_map = {}
    for n, td in agent_td.items():
        for Q, Y, action in zip(td["Q"], td["Y"], td["action"]):
            state = tuple(Q.tolist() + Y.tolist())
            if action[0] == 1: # skip all idling actions
                continue
            if state not in state_action_map:
                state_action_map[state] = []
            state_action_map[state].append(action)
    state_action_maps[(env_num, agent_type, training_env_num)] = state_action_map

# Merge all state action maps if they share the same agent_type and training_env_num
merged_state_action_maps = {}
for (env_num, agent_type, training_env_num), state_action_map in state_action_maps.items():
    if (agent_type, training_env_num) not in merged_state_action_maps:
        merged_state_action_maps[(agent_type, training_env_num)] = {}
    merged_state_action_maps[(agent_type, training_env_num)].update(state_action_map)

# Find where MWN and MLP agent differ if their training_env_num is the same
diffs = {}
for (agent_type, training_env_num), state_action_map in merged_state_action_maps.items():
    if agent_type == "MWN":
        continue
    mwn_state_action_map = merged_state_action_maps[("MWN", training_env_num)]
    diff = {}
    for state, actions in state_action_map.items():
        if state not in mwn_state_action_map:
            continue
        if not (actions[0] == mwn_state_action_map[state][0]).all().item():
            diff[state] = (actions[0].numpy(), mwn_state_action_map[state][0].numpy()) # (MLP, MWN)
    diffs[training_env_num] = diff


# print diff for training environment 0
# Get all of the weights from the MWN agents
mwn_weights = {}
normalized_mwn_weights = {}
env_id_num = 0
for (agent_type, env_id), agent in agent_dict.items():

    if agent_type == "MWN":
        env_id_num += 1
        state_dict = agent.get_policy_operator().state_dict()
        mwn_weights[env_id_num] = state_dict['module.0.module.weights'].detach().numpy()
        normalized_mwn_weights[env_id_num] = mwn_weights[env_id_num]/mwn_weights[env_id_num].sum(axis=0, keepdims=True)

mwn_0_weights = normalized_mwn_weights[1]



# Prepare the headers for the table
headers = ["Q", "Y", "W*Q*Y", "MWN Action", "MLP Action"]

# Prepare the data for the table
table_data = []
for state, (mlp_action, mwn_action) in diffs[0].items():
    Q = state[:5]
    Y = state[5:]
    # compute elemtwise product of Q, Y and W
    WQY = np.round(mwn_0_weights*Q*Y, 2)
    table_data.append([Q, Y, WQY, mwn_action[1:], mlp_action[1:]])

# Print the table
print(tabulate(table_data, headers, tablefmt="pretty"))


# 