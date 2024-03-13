import os
from torchrl_development.envs.sampling.env_sampling import sample_env_params
import json

if __name__ == "__main__":
    # Settings for the sampling loop
    num_admissible_envs = 10
    num_rollouts_per_env = 3
    rollout_steps =  100_000

    # File Loading and Saving
    base_param_file = "SH2.json"
    save_path = f"{base_param_file.split('.')[0]}_generated_params.json"
    # check if save_path exists, if so, then add a number to the end of the file
    if os.path.exists(save_path):
        i = 1
        while os.path.exists(f"{save_path.split('.')[0]}_{i}.json"):
            i += 1
        save_path = f"{save_path.split('.')[0]}_{i}.json"

    # Sample the environment parameters
    multi_env_params = sample_env_params(base_param_file,
                                         num_admissible_envs,
                                         num_rollouts_per_env,
                                         rollout_steps,
                                         keep_dominated=True)

    # Save environment parameters to a file
    with open(save_path, "w") as f:
        json.dump(multi_env_params, f)
    print(f"Saved multi_env_params to {save_path}")