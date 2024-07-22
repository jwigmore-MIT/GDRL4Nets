import numpy as np
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.shortest_queue import ShortestQueueActor
from torchrl_development.envs.env_generators import make_env, parse_env_json
from torchrl_development.utils.metrics import compute_lta
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from torchrl_development.utils.configuration import make_serializable
from datetime import datetime
import torch

"""
Want to sample randomly from a distribution over arrival rates and service rates

"""

def plot_sample(mean_lta, arrival_rates, service_rates, id):
    """
    Plot the lta backlog of the td
    """
    lta_backlog = mean_lta
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(lta_backlog)
    ax.set_title(f"ID {id},Arrival Rates: {arrival_rates}, Service Rates: {service_rates}")



    plt.show()

# Define the environment parameters
base_params = "MP5"
env_params = parse_env_json(f"{base_params}.json")
env_type = env_params["env_type"]
K = env_params["servers"]
lambda_max = 3
lambda_min=2
mu_max = 1


# Define the range of arrival rates and service rates
# lambda_range = np.linspace(lambda_min, lambda_max, 101)# arrival rates
mu_range = np.linspace(0, mu_max, 101) # service rates

# Define the number of samples to take
num_samples = 100
valid_samples = 0
invalid_samples = 0
terminal_backlog = 300
rollout_length = 50000
num_rollouts = 3

# Create MaxWeight Actor
sq_actor = ShortestQueueActor(in_keys=["Q"], out_keys=["action"])

# Create dicitonary to store the samples
context_set_name = f"{base_params}_context_set_l{lambda_max}_m{mu_max}_s{num_samples}.json"
contexts = {}
ltas = []
pbar = tqdm(total=num_samples)
while valid_samples < num_samples:
    pbar.set_postfix({"valid_samples": valid_samples, "invalid_samples": invalid_samples})
    # Generate random arrival rates and service rates
    arrival_rates = np.array([1])
    service_rates = np.random.choice(mu_range, size=K)
    if arrival_rates > service_rates.sum():
        continue
    new_env_params = deepcopy(env_params)
    new_env_params["X_params"]["arrival_rate"] = arrival_rates

    for i in range(K):
        new_env_params["Y_params"][str(i+1)]["service_rate"] = service_rates[i]
    env_ltas = []
    for i in range(num_rollouts):
        env = make_env(new_env_params,
                     observe_lambda=False,
                     seed=i,
                     terminal_backlog=terminal_backlog,
                     observation_keys=["Q", "Y"],
                     inverse_reward=False,
                     stat_window_size=100000,
                     terminate_on_convergence=True,
                     convergence_threshold=0.001,
                     terminate_on_lta_threshold=False)

        td = env.rollout(policy = sq_actor, max_steps=rollout_length)
        if td["next", "backlog"][-1] >= terminal_backlog:
            invalid_samples+=1
            break


        else:
            env_ltas.append(compute_lta(td["next", "backlog"]))

    if len(env_ltas) < num_rollouts:
        continue
    else:
        valid_samples+=1
        pbar.update(1)
        # make all ltas the same length by padding with the last value until they are the same
        max_length = max([len(lta) for lta in env_ltas])
        for i in range(len(env_ltas)):
            env_ltas[i] = np.pad(env_ltas[i], (0, max_length-len(env_ltas[i])), mode='constant', constant_values=(env_ltas[i][-1], env_ltas[i][-1]))
        mean_lta = np.stack(env_ltas).mean(axis = 0)
        ltas.append(mean_lta[-1])
        contexts[str(valid_samples-1)] = {"arrival_rates": arrival_rates, "service_rates": service_rates,
                                        "env_params": new_env_params, "lta": mean_lta[-1]}
        # plot_sample(mean_lta, arrival_rates, service_rates, valid_samples)

json_dict = {"context_dicts": contexts, "ltas": ltas, "num_envs": num_samples}

serial_context_set_dictionary = make_serializable(json_dict)

with open(context_set_name, "w") as f:
    json.dump(serial_context_set_dictionary, f)











