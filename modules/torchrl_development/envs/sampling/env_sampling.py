from torchrl_development.envs.env_generator import parse_env_json, make_env
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import tqdm
import json
import os

"""
Environment sampling without plotting
"""


def remove_dominated_arrays(arrays):
    """
    Remove arrays from the list that are element-wise less than another array in the list.

    Parameters:
    - arrays: List of np.array, each with shape (N,)

    Returns:
    - A list of np.array, filtered as per the criteria.
    """
    # Initialize a list to keep track of indices to remove
    indices_to_remove = set()

    # Compare each array with every other array
    for i, arr_i in enumerate(arrays):
        for j, arr_j in enumerate(arrays):
            if i != j and np.all(arr_i < arr_j):
                indices_to_remove.add(i)

    # Remove the dominated arrays by creating a new list excluding the indices to remove
    filtered_arrays = [arr for i, arr in enumerate(arrays) if i not in indices_to_remove]

    return filtered_arrays, indices_to_remove

def check_dominated(stored_arrays, new_array):
    """
    Check if the new_array is dominated by any of the stored_arrays
    """
    for arr in stored_arrays:
        if np.all(new_array < arr):
            return True
    return False

def remove_dominated_params(param_dict):
    """
    Removes all dict items that have an arrival rate that is elementwise less than another arrival rate in the dict
    :param param_dict:
    :return:
    """
    all_arrival_rates = [param_dict[i]["arrival_rates"] for i in param_dict.keys()]
    non_dom_arrival_rates, removed_indices = remove_dominated_arrays(all_arrival_rates)
    new_dict = {k: v for k, v in param_dict.items() if int(k) not in removed_indices}
    return new_dict



def generate_new_params(base_params, load_factor = None):
    """
    Takes in an environment param files and samples a new set of arrival rates
    :param base_params:
    :return: new_params
    """
    new_params = deepcopy(base_params)
    if "arrival_distribution" in new_params:
        arrival_distribution = new_params["arrival_distribution"]
    else:
        arrival_distribution = "discrete"
        raise NotImplementedError("Need to implement the arrival_distribution = 'discrete' case")


    service_rates = np.array([params["service_rate"] for key, params in new_params["Y_params"].items()])
    if load_factor is None:
        load_factor = np.random.uniform(1, 10)
    U = np.random.uniform(0, 1, size = len(service_rates))
    U = U/np.sum(U)*load_factor
    arrival_rates = U*service_rates

    # truncate arrival rates to 3 decimal place
    arrival_rates = np.trunc(arrival_rates*10**3)/10**3

    for i, params in new_params["X_params"].items():
        ind = int(i)-1
        params["arrival_rate"] = arrival_rates[ind]


    #
    # for node_index, x_params in new_params["X_params"].items():
    #     if arrival_distribution == "discrete":
    #         raise NotImplementedError("Need to red")
    #         arrival_rates.append(np.round(np.random.uniform(0.1, 0.9),1))
    #         x_params["probability"] = arrival_rates[-1]
    #     elif arrival_distribution == "poisson":
    #         if load_factor is not None:
    #             service_rates = np.array([params["service_rate"] for key, params in new_params["Y_params"].items()])
    #             U = np.random.uniform(0, 1, size = len(service_rates))
    #             U = U/np.sum(U)*load_factor
    #             ar = U*service_rates
    #             arrival_rates.append(ar)
    #
    #         arrival_rates.append(np.round(np.random.uniform(1, x_params["arrival_rate"]),0))
    return new_params, arrival_rates

# import and intialize environment



## Implementing all of this as a function
"""
Want to input just the base_param_file, the number of samples, and the rollout_steps
We want to return an ordered dictionary of the key_params, sorted by the lta 


We then want to output the key_params and key_ltas and statistics such as the number of admissible rates, 
    the number of key rates, etc. 
    

"""

def sample_env_params(base_param_file,
                      total_admissible_envs,
                      num_rollouts_per_env,
                      rollout_steps,
                      keep_dominated = True,
                      load_factor = None,
                      terminal_backlog = 250):
    env_params = parse_env_json(base_param_file)
    param_dict = {} # will store the parameters and the lta in a dictionary for each set of arrival rates
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

    # Loop with plotting
    pbar = tqdm.tqdm(total=total_admissible_envs)
    all_env_params = []
    num_admissible = 0
    num_sampled = 0
    sampled_arrival_rates = []
    while num_admissible < total_admissible_envs:

        i = num_sampled
        #update pbar to same num_admissible / num_sampled
        pbar.set_description(f"Admissible/Sampled environments: {num_admissible}/{num_sampled}")
        new_params, arrival_rates = generate_new_params(env_params, load_factor = load_factor)
        if num_admissible > 0:
            if check_dominated(sampled_arrival_rates, arrival_rates):
                continue

        param_dict[i] = {"env_params": new_params,
                                            "arrival_rates": arrival_rates,
                                            "network_load": None,
                                            "lta": None,
                                            "admissible": None,
                                            }
        # get num_envs rollouts per environment to estimate the lta backlog better
        sampled_env_backlogs = np.zeros([num_rollouts_per_env, rollout_steps])
        for n in range(num_rollouts_per_env):
            env = make_env(new_params,
                           seed = 0,
                           device = "cpu",
                           terminal_backlog = 1000)

            # Test MaxWeightActor
            td = env.rollout(policy=max_weight_actor, max_steps = rollout_steps,)
            # if the length of td is not equal to rollout_steps, then quit the loop, and set the admissible to False
            if len(td["backlog"]) != rollout_steps:
                param_dict[i]["admissible"] = False
                break
            else:
                sampled_env_backlogs[n] = td["backlog"].squeeze()
                param_dict[i]["admissible"] = True
                sampled_arrival_rates.append(arrival_rates)
                if n == num_rollouts_per_env - 1:
                    num_admissible += 1
            num_sampled += 1




        # plot the lta_backlogs
        if param_dict[i]["admissible"]:
            ltas = np.array([compute_lta(backlogs) for backlogs in sampled_env_backlogs])
            lta = np.mean(ltas, axis = 0)
            # convert last element of lta to a float
            lta_final: float = lta[-1]
            pbar.update(1)

            # add all extra information into the param_dict[i]
            param_dict[i]["lta"] = lta_final
        else:
            param_dict[i]["lta"] = None
        param_dict[i]["network_load"] = env.network_load
    # Update pbar desc to say getting the admissible rates

    # Get all values of the arrival rates, and if any arrival rate is elementwise less than another arrival rate, then remove it
    admissible_keys = [np.array(k) for k in param_dict.keys() if param_dict[k]["admissible"]]
    admissible_param_dict = {k: param_dict[k] for k in param_dict.keys() if param_dict[k]["admissible"]}
    if not keep_dominated:
        #raise NotImplementedError("Need to implement the keep_dominated = False case")
        pbar.set_description("Getting non-dominated (key) rates")
        admissible_param_dict = remove_dominated_params(admissible_param_dict)
        # key_arrival_rates = [(key, param_dict["arrival_rates"]) for (key,param_dict) in admissible_param_dict.items()]
        # arrival_rates_key_dict = {arrival_rate: key for key, arrival_rate in key_arrival_rates}
        # key_rates = remove_dominated_arrays(admissible_keys)
        # # sort all admisssible keys by the network load
        # key_rates = sorted(key_rates, key = lambda x: param_dict[tuple(x)]["network_load"])


    # Sort admisssible_param_dict by the network load
    admissible_param_dict = {k: v for k, v in sorted(admissible_param_dict.items(), key = lambda x: x[1]["network_load"])}

    # Reindex the admissible_param_dict
    admissible_param_dict = {str(new_i): admissible_param_dict[i] for new_i,i in enumerate(list(admissible_param_dict.keys()))}

    # Get a list of all loads in the admissible_param_dict
    admissible_loads = [admissible_param_dict[i]["network_load"] for i in admissible_param_dict.keys()]

    # get a list of all ltas
    admissible_ltas = [admissible_param_dict[i]["lta"] for i in admissible_param_dict.keys()]

    # count the number of admissible parameters
    num_admissible = len(admissible_param_dict)

    # Create new dictionary
    multi_env_params = {"num_envs": num_admissible,
                        "ltas": admissible_ltas,
                        "network_loads": admissible_loads,
                        "env_params": admissible_param_dict}


    # print the statistics
    print(f"Number of sampled environments: {num_sampled}")
    print(f"Number of admissible rates: {num_admissible}")
    #print(f"Number of key rates: {len(key_rates)}")

    return multi_env_params

def write_env_params_to_file(save_path,key_params, key_ltas, num_admissible, num_key):
    """
    Write the key_params to a json_file, with the key_ltas, and the statistics
    """
    # convert all keys to appropriate types
    key_params = {str(k): v for k, v in key_params.items()}

    with open(save_path, "w") as f:
        json.dump({"num_key": num_key,
                        "key_ltas": key_ltas,
                        "key_params": key_params,}, f)
    print(f"Saved key_params to {save_path}")



"""
Need a function that gets a list of env_params as inputs, runs max weight, and creates a multi_env_params dictionary
like the one above
"""

def get_multi_env_params(env_params_list, rollout_steps = 100_000):
    """
    Takes in a list of env_params, and returns a multi_env_params dictionary
    """
    multi_env_params = {}
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    for i, env_params in enumerate(env_params_list):
        lta_backlogs = []
        for j in range(3):
            env = make_env(env_params,
                           seed = 0,
                           device = "cpu",
                           terminal_backlog = 200)

            # Test MaxWeightActor
            td = env.rollout(policy=max_weight_actor, max_steps = rollout_steps,)

            # plot the lta_backlogs
            backlogs = td["backlog"]
            lta = compute_lta(backlogs)
            # convert last element of lta to a float
            lta_final: float = lta[-1].numpy().tolist()
            lta_backlogs.append(lta_final)
        average_lta_final = np.mean(lta_backlogs)
        std_lta_final = np.std(lta_backlogs)

        # add all extra information into the param_dict[i]
        multi_env_params[i] = {"env_params": env_params,
                                            "arrival_rates": env_params["X_params"],
                                            "network_load": env.network_load,
                                            "lta": (average_lta_final, std_lta_final),
                                            "admissible": lta.shape[0] == rollout_steps,
                                            }
    return multi_env_params


if __name__ == "__main__":
    # Settings for the sampling loop
    num_samples = 500
    rollout_steps = 1000

    # File Loading and Saving
    base_param_file = "SH1.json"
    save_path = f"{base_param_file.split('.')[0]}_generated_params.json"
    # check if save_path exists, if so, then add a number to the end of the file
    if os.path.exists(save_path):
        i = 1
        while os.path.exists(f"{save_path.split('.')[0]}_{i}.json"):
            i += 1
        save_path = f"{save_path.split('.')[0]}_{i}.json"

    # Sample the environment parameters
    multi_env_params = sample_env_params(base_param_file, num_samples, rollout_steps, )

    # Save environment parameters to a file
    with open(save_path, "w") as f:
        json.dump(multi_env_params, f)
    print(f"Saved multi_env_params to {save_path}")

