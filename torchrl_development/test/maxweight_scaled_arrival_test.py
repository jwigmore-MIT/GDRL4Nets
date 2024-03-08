
import matplotlib.pyplot as plt
from torchrl.envs.transforms import CatTensors, TransformedEnv, SymLogTransform, Compose, RewardSum, RewardScaling, StepCounter, ActionMask
import json
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type


from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.env_generator import parse_env_json
from copy import deepcopy
from torchrl_development.utils.configuration import load_config
from torchrl_development.maxweight import MaxWeightActor
import numpy as np
from torchrl_development.utils.metrics import compute_lta
from torchrl_development.envs.env_generator import create_scaled_lambda_generator


env_params = json.load(open("../config/environments/SH1_NA.json", 'rb'))["problem_instance"]
lambda_scales = [0.98, 0.95, 0.9, 0.85, 0.8]

device = "cpu"
make_env_parameters = {"observe_lambda": False,
                       "device": device,
                       }
results = {}

for lambda_scale in lambda_scales:

    env_generator = create_scaled_lambda_generator(env_params,
                                                   make_env_parameters,
                                                   env_generator_seed=0,
                                                   lambda_scale=lambda_scale)
    # Define the environment
    env = env_generator.sample()
    # Run check env
    check_env_specs(env)

    # Create actor network
    input_shape = env.observation_spec["observation"].shape
    output_shape = env.action_spec.space.n
    # distribution = MaskedOneHotCategorical
    max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    all_lta_backlogs = []
    max_effective_q_space = 0
    for i in range(3):
        test_env = env_generator.sample()
        td = test_env.rollout(policy=max_weight_actor, max_steps = 50000)
        backlogs = td["backlog"]
        effective_q_space = td["Q"].max(axis = 0).values.prod()
        if effective_q_space > max_effective_q_space:
            max_effective_q_space = effective_q_space
        lta_backlogs = compute_lta(backlogs.numpy())
        all_lta_backlogs.append(lta_backlogs)
        # np.divide(backlogs.numpy().sum(),np.arange(1, len(backlogs)+1).reshape(-1,1)))
    # plot backlogs
    results[lambda_scale] = {"lta":np.array(all_lta_backlogs),
                             "effective_q_space": max_effective_q_space.numpy()}
fig, ax = plt.subplots()
for key in results.keys():
    mean_lta_backlogs = np.mean(results[key]["lta"], axis=0)
    std_lta_backlogs = np.std(results[key]["lta"], axis=0)
    ax.plot(mean_lta_backlogs, label=f"lambda_scale = {key}")
    ax.fill_between(np.arange(len(mean_lta_backlogs)), mean_lta_backlogs-std_lta_backlogs, mean_lta_backlogs+std_lta_backlogs, alpha=0.5)


ax.set(xlabel='time', ylabel='backlog',
       title='MaxWeight LTA Backlog over time')
ax.grid()
fig.legend()
fig.show()






