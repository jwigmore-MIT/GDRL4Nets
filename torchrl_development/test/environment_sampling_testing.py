from torchrl_development.envs.env_generator import parse_env_json, make_env
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np


def generate_new_params(base_params):
    """
    Takes in an environment param files and samples a new set of arrival rates
    :param base_params:
    :return: new_params
    """
    new_params = deepcopy(base_params)
    arrival_rates = []
    for node_index, x_params in new_params["X_params"].items():
        arrival_rates.append(np.round(np.random.uniform(0.1, 0.9),1))
        x_params["probability"] = arrival_rates[-1]
    return new_params, arrival_rates

# import and intialize environment
env_name = "SH2.json"
env_params = parse_env_json(env_name)
param_dict = {} # will store the parameters and the lta in a dictionary for each set of arrival rates
max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
rollout_steps = 1000

# Loop with plotting
for i in range(10):
    new_params, arrival_rates = generate_new_params(env_params)
    param_dict[tuple(arrival_rates)] = {"env_params": new_params,
                                        }
    env = make_env(new_params,
                   max_steps = rollout_steps,
                   seed = 0,
                   device = "cpu",
                   terminal_backlog = 100)

    # Test MaxWeightActor
    td = env.rollout(policy=max_weight_actor, max_steps = rollout_steps,)

    # plot the lta_backlogs
    backlogs = td["backlog"]
    lta = compute_lta(backlogs)
    param_dict[tuple(arrival_rates)]["lta"] = lta[-1]
    param_dict[tuple(arrival_rates)]["passed"] = lta.shape[0] == rollout_steps

    if param_dict[tuple(arrival_rates)]["passed"]:
        fig, ax = plt.subplots()
        ax.plot(lta)
        ax.set(xlabel='time', ylabel='lta',
               title=f"Arrival Rates: {arrival_rates}")
        ax.grid()
        plt.show()






