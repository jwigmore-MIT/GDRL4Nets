
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
lambda_scale = 0.98

device = "cpu"
make_env_parameters = {"observe_lambda": False,
                       "device": device,
                       }

env_generator = create_scaled_lambda_generator(env_params,
                                               make_env_parameters,
                                               env_generator_seed=531997,
                                               lambda_scale=lambda_scale)
# Define the environment
env = env_generator.sample()
# Run check env
check_env_specs(env)

# distribution = MaskedOneHotCategorical
max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
all_lta_backlogs = []
for i in range(3):
    test_env = env_generator.sample()
    td = test_env.rollout(policy=max_weight_actor, max_steps = 50_000)
    backlogs = td["backlog"]
    lta_backlogs = compute_lta(backlogs.numpy())
    all_lta_backlogs.append(lta_backlogs)
    # np.divide(backlogs.numpy().sum(),np.arange(1, len(backlogs)+1).reshape(-1,1)))
# plot backlogs
ltas = np.array(all_lta_backlogs)
mean_lta_backlogs = np.mean(all_lta_backlogs, axis=0)
std_lta_backlogs = np.std(all_lta_backlogs, axis=0)
# 95% confidence interval
ci = std_lta_backlogs*1.96/np.sqrt(len(all_lta_backlogs))
fig, ax = plt.subplots()
ax.plot(mean_lta_backlogs)
ax.fill_between(np.arange(len(mean_lta_backlogs)), mean_lta_backlogs-ci, mean_lta_backlogs+ci, alpha=0.5)
ax.set(xlabel='time', ylabel='backlog',
       title='LTA Backlog over time')
ax.grid()
plt.show()


fig, ax = plt.subplots()
ax.plot(ltas.T)
ax.set(xlabel='time', ylabel='backlog',
       title='LTA Backlog over time')
ax.grid()
plt.show()


