
# import make_scaled_gen_from_params from env_sampling
from torchrl_development.envs.sampling.env_sampling import get_multi_env_params
# import create_scaled_lambda_params from env_generator
from torchrl_development.envs.env_generator import create_scaled_lambda_params

import json

# Get SH1_NA, SH1_NA12, SH1_NA2, SH1_NA13, and SH1_NA3 params
SH1_NA_params = json.load(open(f"../../config/environments/SH1_NA.json", 'rb'))["problem_instance"]
SH1_NA12_params = json.load(open(f"../../config/environments/SH1_NA12.json", 'rb'))["problem_instance"]
SH1_NA2_params = json.load(open(f"../../config/environments/SH1_NA2.json", 'rb'))["problem_instance"]
SH1_NA13_params = json.load(open(f"../../config/environments/SH1_NA13.json", 'rb'))["problem_instance"]
SH1_NA3_params = json.load(open(f"../../config/environments/SH1_NA3.json", 'rb'))["problem_instance"]


# scale the lambda of the SH1_NA_params by 0.
scaled_SH1_NA_params0 = create_scaled_lambda_params(SH1_NA_params, 0.95)
scaled_SH1_NA_params1 = create_scaled_lambda_params(SH1_NA_params, 0.85)
scaled_SH1_NA_params2 = create_scaled_lambda_params(SH1_NA_params, 0.98)

# For the remaining params, scale the lambda by 0.95
scaled_SH1_NA12_params = create_scaled_lambda_params(SH1_NA12_params, 0.95)
scaled_SH1_NA2_params = create_scaled_lambda_params(SH1_NA2_params, 0.95)
scaled_SH1_NA13_params = create_scaled_lambda_params(SH1_NA13_params, 0.95)
scaled_SH1_NA3_params = create_scaled_lambda_params(SH1_NA3_params, 0.95)

# Now create a multi_env_params dictionary using the above params
#env_params_list = [scaled_SH1_NA_params0, scaled_SH1_NA_params1, scaled_SH1_NA_params2, scaled_SH1_NA12_params, scaled_SH1_NA2_params, scaled_SH1_NA13_params, scaled_SH1_NA3_params]
env_params_list = [scaled_SH1_NA_params0]
print("Creating multi_env_params")
multi_env_params = get_multi_env_params(env_params_list, rollout_steps=500_000)
gen_params = {"all_env_params": multi_env_params,
                "num_envs": len(env_params_list)}

# Create generator as a test
from torchrl_development.envs.env_generator import EnvGenerator

make_env_parameters = {"observe_lambda": False,
                          "device": "cpu",
                          "terminal_backlog": 250,
                          "inverse_reward": True,
                          }

env_gen = EnvGenerator(gen_params,
                          make_env_parameters,
                          env_generator_seed=0)

all_sampled_envs = env_gen.create_all_envs()


# Save environment parameters to a file
with open("sampled_envs_json\\experiment4b_envs_params.json", "w") as f:
    json.dump(gen_params, f)


