from torchrl_development.trainer import train
import json
from torchrl_development.envs.env_generator import create_scaled_lambda_generator, EnvGenerator
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_actor_critic
from datetime import datetime
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
import numpy as np
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
import torch
from torchrl.record.loggers import get_logger
import os
import yaml
import wandb
import time
from tensordict import TensorDict
from copy import deepcopy
from tqdm import tqdm
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type




cfg = load_config(full_path= "experiment4.yaml")
all_env_params = json.load(open("experiment4_envs_params.json", 'rb'))
input_params = {"num_envs": len(all_env_params.keys()),
                "all_env_params": all_env_params,}

# creating eval env_generators
make_env_parameters = {"observe_lambda": cfg.agent.observe_lambda,
                            "device": cfg.device,
                            "terminal_backlog": cfg.eval_envs.terminal_backlog,
                            "inverse_reward": cfg.eval_envs.inverse_reward,
                            }

gen_env_generator = EnvGenerator(input_params,
                                 make_env_parameters,
                                env_generator_seed=1010110)








# # Create base env for agent generation
base_env= gen_env_generator.sample(0)
check_env_specs(base_env)
#
# # Create agent
input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n
#
agent = create_actor_critic(
    input_shape,
    output_shape,
    in_keys=["observation"],
    action_spec=base_env.action_spec,
    temperature=cfg.agent.temperature,
)

# Set device
device = cfg.device

#  Specify actor and critic
actor = agent.get_policy_operator().to(device)

# create MaxWeightActor
max_weight_actor = MaxWeightActor()

env_indices = [0,1,3,5]

# test the actor on each env in env_indices
# and test the maxweight policy on each env in env_indices
results = {}
with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
    for ind in env_indices:
        results[ind] = {}
        for agent, name in zip([actor, max_weight_actor], ["randomized_actor", "max_weight_actor"]):
            lta_backlog_list = []
            for n in range(3):
                env = gen_env_generator.sample(ind)
                print(f"Testing {name} on env {ind}")
                td = env.rollout(max_steps = 50_000, policy = agent)
                backlogs = td["backlog"]
                lta_backlogs = compute_lta(backlogs)
                lta_backlog_list.append(lta_backlogs)
            results[ind][name] = lta_backlog_list

# Plot the results by plotting the mean of the lta_backlogs for each ind and name
# and the std of the lta_backlogs for each ind and name
# label each line with the name+ind
# use a different linestyle for each name
# the color should be the same for each agent on the same env_id

# create a list of colors the same length as the number of env_ind
colors = ["r", "b", "g", "y"]
color_dict = {ind: color for ind, color in zip(env_indices, colors)}

fig, ax = plt.subplots()
for ind in env_indices:
    for name in ["randomized_actor", "max_weight_actor"]:
        data = np.array(results[ind][name])
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # if name is "randomized_actor": linestyle = "--"
        # else: linestyle = "-"
        linestyle = "--" if name == "randomized_actor" else "-"
        x = np.arange(len(mean))
        ax.plot(mean, label=f"{name} on env {ind}", linestyle=linestyle, color = color_dict[ind])
        ax.fill_between(x, mean-std, mean+std, color = color_dict[ind], alpha=0.2)
ax.legend()
plt.show()


