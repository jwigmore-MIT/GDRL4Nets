import torch
import pickle
import os
from tqdm import tqdm
import argparse
import json

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type


from modules.torchrl_development.envs.env_creation import EnvGenerator
from modules.torchrl_development.utils.metrics import compute_lta
from modules.torchrl_development.agents.max_weight import create_maxweight_actor_critic
from modules.torchrl_development.agents.actors import create_actor_critic
from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.agents.actors import create_independent_actor_critic


parser = argparse.ArgumentParser(description='Test Agents')
# add argument for context set folder
parser.add_argument('--training_set_folder', type=str, help='folder containing agents trained according to a particular training set', default="SH4_0-5_b")
# add argument for context set file name
parser.add_argument('--context_set_file_name', type=str, help='file name of context set', default="SH4_context_set_l3_m3_s100.json")
# add argument for agent types (list of strings)
parser.add_argument('--agent_types', nargs='+', type=str, help='list of agent types', default=["MLP"])

args = parser.parse_args()


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


training_set_folder = args.training_set_folder
context_set_file_name = args.context_set_file_name
agent_types = args.agent_types



rollout_length = 30000
num_rollouts = 3
env_generator_seed = 4162024
lta_tds = {}
trained_agent_folder = os.path.join("trained_agents",training_set_folder)
test_context_set_path = os.path.join("context_sets", context_set_file_name)
context_set = json.load(open(test_context_set_path, 'rb'))
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


pmn_cfg = load_config(os.path.join(SCRIPT_PATH, 'PMN_Shared_PPO_settings.yaml'))
mlp_cfg = load_config(os.path.join(SCRIPT_PATH, 'MLP_PPO_settings.yaml'))



make_env_parameters = {"graph": False,
                                    "observe_lambda": True,
                                    "observe_mu": True,
                                    "terminal_backlog": None,
                                    "observation_keys": ["Q", "Y"],
                                    "observation_keys_scale": None,
                                    "negative_keys": None,
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
                                                actor_in_keys=["Q", "Y", "lambda"],
                                                critic_in_keys=["observation"],
                                                action_spec=action_spec,
                                                temperature=pmn_cfg.agent.temperature,
                                                actor_depth=pmn_cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=pmn_cfg.agent.hidden_sizes[-1],
                                                type=3,
                                                network_type="PMN",
                                                relu_max=getattr(pmn_cfg, "relu_max", 10), )

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



