import os
from torchrl_development.envs.sampling.context_space_sampling import sample_context_space
from torchrl_development.utils.configuration import make_serializable
import json

if __name__ == "__main__":
    # Settings for the sampling loop
    num_admissible_envs = 30
    num_rollouts_per_env = 1
    rollout_steps =  10000
    keep_dominated = False
    load_factor = 1.32

    # File Loading and Saving
    base_param_file = "SH2u.json"
    if keep_dominated:
        save_path = f"{base_param_file.split('.')[0]}_lf{load_factor}_context_space.json"
    else:
        save_path = f"{base_param_file.split('.')[0]}_lf{load_factor}_context_space-nondominated.json"
    # check if save_path exists, if so, then add a number to the end of the file
    if os.path.exists(save_path):
        i = 1
        while os.path.exists(f"{save_path.split('.')[0]}_{i}.json"):
            i += 1
        save_path = f"{save_path.split('.')[0]}_{i}.json"

    # Sample the environment parameters
    context_space_dict = sample_context_space(base_param_file,
                                           num_admissible_envs,
                                           num_rollouts_per_env,
                                           rollout_steps,
                                           keep_dominated=False,
                                           load_factor=1.32,  #1.32
                                           terminal_backlog=1000)


    serial_context_dictionary = make_serializable(context_space_dict)
    with open(save_path, "w") as f:
        json.dump(serial_context_dictionary, f)
    print(f"Saved multi_env_params to {save_path}")