import torch
import pickle
import os
import sys
from torchrl_development.mdp_actors import MDP_actor, MDP_module
import json
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))



rollout_length = 10000
num_rollouts = 3

results = {}
test_context_set_path = 'SH1_context_set.json'
mdp_dir_rel_path = "MDP_Solver/saved_mdps/"
mdp_dir = os.path.join(PROJECT_DIR, mdp_dir_rel_path)
# load all pickled agents from mdp_dir
mdps = {}
for file in os.listdir(mdp_dir):
    if file.endswith(".p"):
        mdp = pickle.load(open(os.path.join(mdp_dir, file), 'rb'))
        mdps[file] = mdp

# Create MDP agents
mdp_agents = {}
for env_id, (mdp_name, mdp) in enumerate(mdps.items()):
    mdp_module = MDP_module(mdp)
    mdp_agents[env_id] = MDP_actor(mdp_module, in_keys = ["Q", "Y"], out_keys = ["action"])

# Load Environments
context_set_path = 'SH1_context_set.json'
test_context_set = json.load(open(context_set_path, 'rb'))
make_env_parameters = {"observe_lambda": False,
                           "device": "cpu",
                           "terminal_backlog": 5000,
                           "inverse_reward": True,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": False,
                           "convergence_threshold": 0.1,
                           "terminate_on_lta_threshold": False, }
env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=111)


# Test each agent on each environment
results = {}
for env_id in range(env_generator.num_envs):
    for agent_id, agent in mdp_agents.items():
        env = env_generator.sample(env_id)
        ltas = []
        for n in range(num_rollouts):
            env.reset()
            td = env.rollout(policy = agent, max_steps = rollout_length)
            ltas.append(compute_lta(td["backlog"]))
        results[(env_id, agent_id)] = torch.stack(ltas)

# plot the mean with confidence intervals for each (env_id, agent_id) pair in results
# create a new plot for each environment

for env_id in range(env_generator.num_envs):
    fig, ax = plt.subplots()
    for agent_id in mdp_agents.keys():
        max_weight_lta = test_context_set["context_dicts"][str(env_id)]["lta"]
        ltas = results[(env_id, agent_id)]
        mean = ltas.mean(dim=0)/max_weight_lta
        std = ltas.std(dim=0)/(max_weight_lta**2)
        ax.plot(mean, label=f"Agent: {agent_id}")
        ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2)
    ax.legend()
    ax.set_title(f"Environment {env_id}")
    plt.show()


