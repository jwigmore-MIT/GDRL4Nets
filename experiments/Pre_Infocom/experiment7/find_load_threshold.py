# test to see if poisson arrival/service rates work as intended

# import make_env
import numpy as np
from torchrl_development.maxweight import MaxWeightActor
from torchrl_development.envs.env_generators import make_env, parse_env_json
from torchrl_development.utils.metrics import compute_lta
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm

def find_load_threshold(env_params, load_range, repeats = 3, seed=0, terminal_backlog = 1000):
    """
    Find the load threshold for a given environment parameters and load
    """

    # create a pbar that will be used to track the progress of the experiment
    pbar = tqdm(total=len(load_range)*repeats)
    final_lta_backlogs= []
    admissible_count = []
    for load in load_range:
        ltas = []
        adm_count = 0
        for trial in range(repeats):
            # Sample a uniform rv of length equal to the number of X_params
            pbar.set_postfix({"load": load, "trial": trial})
            new_params = deepcopy(env_params)

            # get Y_params
            Y_params = new_params["Y_params"]

            # get the service rates as a numpy array from the Y_params
            service_rates = np.array([params["service_rate"] for key, params in Y_params.items()])
            U = np.random.uniform(0, 1, size = len(Y_params))
            # Normalize the uniform rv
            U = U/np.sum(U)*load
             # Compute new arrival rates by mulitplying U by the service rates
            arrival_rates = U*service_rates

            # Set new_params arrival rates to the new arrival rates
            for i, params in new_params["X_params"].items():
                ind = int(i)-1
                params["arrival_rate"] = arrival_rates[ind]

            for i, params in new_params["Y_params"].items():
                ind = int(i)-1
                params["service_rate"] = service_rates[ind]

            env = make_env(new_params,
                 observe_lambda = False,
                 seed=seed,
                 device="cpu",
                 terminal_backlog=terminal_backlog,
                 observation_keys=["Q", "Y"],
                 inverse_reward= False,
                 stat_window_size =  100000,
                 terminate_on_convergence =  True,
                 convergence_threshold = 0.1,
                 terminate_on_lta_threshold =  False,
            )

            max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
            # print("Collecting Max Weight rollout")
            td = env.rollout(policy=max_weight_actor, max_steps = 50000)
            if td["next", "backlog"][-1] >= terminal_backlog:
                pass
            else:
                adm_count += 1
            lta = compute_lta(td["backlog"])
            ltas.append(lta[-1])
            pbar.update(1)
        admissible_count.append(adm_count)
        final_lta_backlogs.append(np.mean(ltas))

    #plot load vs final_lta_backlogs on first y axis and admissible count on second y axis
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(load_range, final_lta_backlogs, 'g-')
    ax2.plot(load_range, admissible_count, 'b-')
    ax.set_xlabel('Load')
    ax.set_ylabel('Final LTA Backlog', color='g')
    ax2.set_ylabel('Admissible Count', color='b')
    plt.title("Load vs Final LTA Backlog and Admissible Count")
    plt.show()


    plt.plot(load_range, final_lta_backlogs)
    plt.xlabel("Load")
    plt.ylabel("Avg Final LTA Backlog")
    service_rates = np.array([params["service_rate"] for key, params in Y_params.items()])
    plt.title(f"Load vs Avg Final LTA Backlog \n Service Rates: {service_rates}")

    plt.show()
    return load_range, final_lta_backlogs, admissible_count


if __name__ == "__main__":
    env_params = parse_env_json("SH2u.json")
    service_rates = np.array([params["service_rate"] for key, params in env_params["Y_params"].items()])
    load_range = np.linspace(1.3, 1.4, 10)
    load_range, final_lta_backlogs, admissible_count =find_load_threshold(env_params, load_range, repeats = 3, seed=0)




