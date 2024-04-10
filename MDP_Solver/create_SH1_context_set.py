from torchrl_development.envs.env_generators import make_env, parse_env_json, create_scaled_lambda_params
from torchrl_development.actors import MaxWeightActor
from tqdm import tqdm
from torchrl_development.utils.configuration import make_serializable
import json
import numpy as np
from torchrl_development.utils.metrics import compute_lta
import torch
"""
Main idea is to create a context dictionary from the SH1NA environments.
"""
env_name_list = ["SH1_NA", "SH1_NA2", "SH1_NA3"]
scale_parameters = [0.95]
num_rollouts_per_env = 3
max_steps =50000


def estimate_lta_backlog(env_params,
                         num_rollouts_per_env = 3,
                         max_rollout_steps = 50000,
                         **make_env_kwargs):
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    max_weight_ltas = []
    Q_maxs = []
    for i in range(num_rollouts_per_env):
        env = make_env(env_params, seed = int(i+10),  **make_env_kwargs)
        td = env.rollout(policy=max_weight_actor, max_steps = max_rollout_steps)
        # get largest Q value from the rollout
        Q_maxs.append(td["Q"].max(dim = 0).values)
        if len(td) == max_rollout_steps or td["next"]["terminated"][-1]:
            max_weight_ltas.append(compute_lta(td["backlog"]))
        elif td["next"]["truncated"][-1]:
            print(f"Rollout {i} truncated at step {len(td)}")
            raise ValueError("Rollout truncated")
    Q_max = torch.stack(Q_maxs).max(dim = 0).values.numpy()
    network_load = env.base_env.network_load
    final_ltas = np.array([lta[-1] for lta in max_weight_ltas])
    if True:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,1, figsize = (15,10))
        for i in range(num_rollouts_per_env):
            axes.plot(max_weight_ltas[i], label = f"Rollout {i}")
        axes.set_title(f"MaxWeight LTAs for lambda {np.round(env.base_env.arrival_rates, decimals=2)}" )
        fig.show()

    return np.mean(final_ltas), np.std(final_ltas), network_load, Q_max


max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])


context_set_dict = {}
context_set_dict["context_dicts"] = {}
context_set_dict["num_envs"] = len(env_name_list)
context_set_dict["ltas"] = []
context_set_dict["network_loads"] = []


pbar = tqdm(total=len(env_name_list)*num_rollouts_per_env)
make_env_keywords = {
    "stat_window_size": 100000,
    "terminate_on_convergence": False,
    "convergence_threshold": 0.01

}

for e, env_name in enumerate(env_name_list):
    for n,scale in enumerate(scale_parameters):
        env_params = parse_env_json(f"{env_name}.json")
        scaled_params = create_scaled_lambda_params(env_params, scale)
        context_dict = {}
        env =  make_env(env_params, **make_env_keywords)
        context_dict["env_params"] = scaled_params
        context_dict["arrival_rates"] = env.arrival_rates
        max_weight_lta, max_weight_lta_stdev, network_load, Q_max = estimate_lta_backlog(scaled_params,
                                                                                  num_rollouts_per_env,
                                                                                  max_steps,
                                                                                  **make_env_keywords)
        context_dict["lta"] = max_weight_lta
        context_dict["lta_stdev"] = max_weight_lta_stdev
        context_dict["admissible"] = True
        context_dict["network_load"] = network_load
        context_dict["Q_max"] = Q_max
        context_set_dict["context_dicts"][e] = context_dict
        context_set_dict["ltas"].append(max_weight_lta)
        context_set_dict["network_loads"].append(network_load)
        pbar.update(3)

serial_context_set_dictionary = make_serializable(context_set_dict)
with open("SH1_context_set_095.json", "w") as f:
    json.dump(serial_context_set_dictionary, f)

