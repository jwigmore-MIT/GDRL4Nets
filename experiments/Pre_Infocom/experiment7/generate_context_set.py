import json
import numpy as np
from scipy.spatial import ConvexHull
from diversipy import polytope
import json
from copy import deepcopy
from torchrl_development.envs.env_generators import make_env
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
import tqdm
from datetime import datetime
from context_set_stats import plot_arrival_rate_histogram
from torchrl_development.utils.configuration import make_serializable
from torchrl_development.envs.sampling.polytope_sampling import dikin_walk, collect_chain, chebyshev_center
import os

def sample_contexts_hit_and_run(context_space_dict, num_samples, thin = 100):
    vertex_arrival_rates = np.array([context_space_dict["context_dicts"][str(i)]["arrival_rates"] for i in  range(context_space_dict["num_envs"])])
    # add the zero vector to the vertex_arrival_rates
    vertex_arrival_rates = np.vstack((vertex_arrival_rates, np.zeros(vertex_arrival_rates.shape[1])))

    hull = ConvexHull(vertex_arrival_rates)
    equations = hull.equations
    lower = np.zeros(vertex_arrival_rates.shape[1])
    upper = np.ones(vertex_arrival_rates.shape[1])*10000
    A = equations[:,:-1]
    b = -equations[:,-1]

    samples = polytope.sample(num_samples, lower, upper, A1 =A, b1 = b, thin = 100)
    return samples

def sample_contexts_dilkins(context_space_dict, num_samples):

    vertex_arrival_rates = np.array([context_space_dict["context_dicts"][str(i)]["arrival_rates"] for i in  range(context_space_dict["num_envs"])])
    # add the zero vector to the vertex_arrival_rates
    vertex_arrival_rates = np.vstack((vertex_arrival_rates, np.zeros(vertex_arrival_rates.shape[1])))

    hull = ConvexHull(vertex_arrival_rates)
    equations = hull.equations

    A = equations[:,:-1]
    b = -equations[:,-1]
    x0 = chebyshev_center(A, b)
    count, burn, thin, step_size = 1000, 100, 1, 1
    samples = np.array([collect_chain(dikin_walk, count, burn, thin, A, b, x0, 1)[-1] for _ in range(num_samples)])

    return samples


def estimate_lta_backlog(env_params,
                         num_rollouts_per_env = 3,
                         max_rollout_steps = 50000,
                         **make_env_kwargs):
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    max_weight_ltas = []
    for i in range(num_rollouts_per_env):
        env = make_env(env_params, **make_env_kwargs)
        td = env.rollout(policy=max_weight_actor, max_steps = max_rollout_steps)
        if len(td) == max_rollout_steps or td["next"]["terminated"][-1]:
            max_weight_ltas.append(compute_lta(td["backlog"]))
        elif td["next"]["truncated"][-1]:
            print(f"Rollout {i} truncated at step {len(td)}")
            raise ValueError("Rollout truncated")
    network_load = env.base_env.network_load
    final_ltas = np.array([lta[-1] for lta in max_weight_ltas])
    if False:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1,1, figsize = (15,10))
        for i in range(num_rollouts_per_env):
            axes.plot(max_weight_ltas[i], label = f"Rollout {i}")
        axes.set_title(f"MaxWeight LTAs for lambda {np.round(env.base_env.arrival_rates, decimals=2)}" )
        fig.show()

    return np.mean(final_ltas), np.std(final_ltas), network_load


def create_context_set_dict(context_parameters: np.array,
                            base_env_params: dict,
                            make_env_params: dict):
    N,K = context_parameters.shape
    context_parameters = context_parameters
    base_env_params = deepcopy(base_env_params)
    context_set_dict = {}
    context_set_dict["context_dicts"] = {}
    context_set_dict["num_envs"] = N
    context_set_dict["ltas"] = []
    context_set_dict["network_loads"] = []

    num_rollouts_per_env = 3
    max_rollout_steps = 50000
    pbar = tqdm.tqdm(total=N)
    pbar.set_description(f"Computing LTA for sampled contexts")
    for s, sample in enumerate(context_parameters):
        context_dict = {}
        context_dict["arrival_rates"] = sample

        # create new env_params using arrival rates
        env_params = deepcopy(base_env_params)
        x_params = deepcopy(env_params["X_params"])
        for i, x_params in env_params["X_params"].items():
            x_params["arrival_rate"] = sample[int(i) - 1]
        context_dict["env_params"] = env_params
        max_weight_lta, max_weight_lta_stdev, network_load = estimate_lta_backlog(env_params,
                                                                                  num_rollouts_per_env,
                                                                                  max_rollout_steps,
                                                                                  **make_env_params)
        context_dict["lta"] = max_weight_lta
        context_dict["lta_stdev"] = max_weight_lta_stdev
        context_dict["admissible"] = True
        context_dict["network_load"] = network_load
        context_set_dict["context_dicts"][s] = context_dict
        context_set_dict["ltas"].append(max_weight_lta)
        context_set_dict["network_loads"].append(network_load)
        pbar.update(1)
    return context_set_dict



if __name__ == "__main__":

        base_name = "SH3"
        num_contexts = 100
        thin = 1000
        sampling_method = 'hit_and_run' #'dilkins' or 'hit_and_run'
        date_time = datetime.now().strftime('%m%d%H%M')

        # create folder to store all the sampled context parameters and context set dictionary
        folder_name = f"{base_name}_sampled_contexts_{num_contexts}_{sampling_method}_{date_time}"
        os.mkdir(folder_name)
        param_save_file = f"{base_name}_sampled_context_parameters_{num_contexts}_{sampling_method}_{date_time}.json"
        context_set_dict_file_name = f"{base_name}_context_set_{num_contexts}_{date_time}.json"



        context_space_dict = json.load(open("SH3_lf1.38_context_space-nondominated.json", 'rb'))
        if sampling_method == 'hit_and_run':
            context_samples = sample_contexts_hit_and_run(context_space_dict, num_contexts, thin = thin)
        else:
            context_samples = sample_contexts_dilkins(context_space_dict, num_contexts)
        # plot the context samples
        plot_arrival_rate_histogram(context_samples, title = f"{sampling_method} Sampling Parameters Histogram")
        #plot_arrival_rate_histogram(dikin_context_samples)
        # Save contexts to a json file

        serial_samples = [list(sample) for sample in context_samples]
        with open(os.path.join(folder_name,param_save_file), "w") as f:
            json.dump(serial_samples, f)
        #
        # sampled_context_parameters = json.load(open("SH2u_sampled_context_parameters.json", 'rb'))
        # # for each sample, create a new context dict
        make_env_params = {"observe_lambda": False,
                           "seed": None,
                           "device": "cpu",
                           "terminal_backlog": None,
                           "observation_keys": ["Q", "Y"],
                           "inverse_reward": False,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": True,
                           "convergence_threshold": 0.1}


        context_set_dict = create_context_set_dict(context_samples,
                                                   context_space_dict["context_dicts"]["0"]["env_params"],
                                                   make_env_params)
        serial_context_set_dictionary = make_serializable(context_set_dict)
        with open(os.path.join(folder_name, context_set_dict_file_name), "w") as f:
             json.dump(serial_context_set_dictionary, f)




