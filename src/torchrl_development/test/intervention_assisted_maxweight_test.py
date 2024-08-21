
import matplotlib.pyplot as plt
from torchrl.envs.transforms import CatTensors, TransformedEnv, SymLogTransform, Compose, RewardSum, RewardScaling, StepCounter, ActionMask

from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type


from torchrl_development.envs.SingleHop import SingleHop
from torchrl_development.envs.env_generator import parse_env_json
from copy import deepcopy
from torchrl_development.utils.configuration import load_config
from torchrl_development.maxweight import MaxWeightActor
import numpy as np
from torchrl_development.utils.metrics import compute_lta
from torchrl.modules import Actor, MLP, ProbabilisticActor, ValueOperator, MaskedOneHotCategorical, ActorCriticWrapper
from tensordict.nn import TensorDictModule
from torchrl.data import CompositeSpec
from torchrl_development.intervention_actor import InterventionActor, InterventionActorCriticWrapper
import torch


env_name = "SH2"
env_params = parse_env_json(f"{env_name}.json")
# load config as a namespace
cfg = load_config("ppo1.yaml")
def make_env(env_params = env_params,
             max_steps = 2048,
             seed = 0,
             device = "cpu",
             terminal_backlog = None):
    env_params = deepcopy(env_params)
    if terminal_backlog is not None:
        env_params["terminal_backlog"] = terminal_backlog

    base_env = SingleHop(env_params, seed, device)
    env = TransformedEnv(
        base_env,
        Compose(
            # normalize observations
            ActionMask(action_key="action", mask_key="mask"),
            CatTensors(in_keys=["Q", "Y"], out_key="observation", del_keys=False),
            SymLogTransform(in_keys=["observation"], out_keys=["observation"]),
            #InverseReward(),
            RewardScaling(loc = 0, scale=0.01),
            RewardSum(),
            StepCounter(max_steps = max_steps)
        )
    )
    return env




device = cfg.device

# Define the environment
env = make_env(env_params)

# Run check env
check_env_specs(env)

# Create actor network
input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.space.n
# distribution = MaskedOneHotCategorical
max_weight_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])

# Create actor network
input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.space.n
# distribution = MaskedOneHotCategorical
in_keys= ["observation"]
actor_mlp = MLP(in_features=input_shape[-1],
          activation_class = torch.nn.ReLU,
          activate_last_layer = True,
          out_features=output_shape,
          )
actor_mlp_output = actor_mlp(torch.ones(input_shape))

critic_mlp = MLP(in_features=input_shape[-1],
          activation_class = torch.nn.ReLU,
          activate_last_layer = True,
          out_features=1,
          )
critic_mlp_output = critic_mlp(torch.ones(input_shape))

actor_module = TensorDictModule(
    module = actor_mlp,
    in_keys = in_keys,
    out_keys = ["logits"],
)
# Add probabilistic sampling to the actor
# prob_module = ProbabilisticTensorDictModule(
#     in_keys = ["logits", "mask"],
#     out_keys = ["action", "log_prob"],
#     distribution_class= MaskedOneHotCategorical,
#     default_interaction_type=ExplorationType.RANDOM,
#     return_log_prob=True,
# )

# actor_module = ProbabilisticTensorDictSequential(actor_module, prob_module)


actor_module = ProbabilisticActor(
    actor_module,
    distribution_class= MaskedOneHotCategorical,
    #distribution_kwargs = {"mask_key": "mask"},
    in_keys = ["logits", "mask"],
    spec=CompositeSpec(action=env.action_spec),
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM,
)

ia_actor_module = InterventionActor(actor_module, intervention_type="maxweight", threshold = 30)
# Create Value Module
value_module = ValueOperator(
    module = critic_mlp,
    in_keys = in_keys,
)

#
actor_critic = InterventionActorCriticWrapper(ia_actor_module, value_module)





#test ia + maxweight policy
env = make_env(env_params = env_params,
                 max_steps = 20480,
                 seed = 0,
                terminal_backlog = None)

td = env.rollout(policy=ia_actor_module, max_steps = 20480)

backlogs = td["backlog"]
interventions = td["intervene"]
lta_backlogs = compute_lta(backlogs.numpy())
    # np.divide(backlogs.numpy().sum(),np.arange(1, len(backlogs)+1).reshape(-1,1)))
# plot backlogs
fig, ax = plt.subplots(2,1)
ax[0].plot(lta_backlogs)
ax[0].set(xlabel='time', ylabel='backlog',
       title='LTA Backlog over time')
ax[0].grid()
# plot interventions
ax[1].plot(interventions)
ax[1].set(xlabel='time', ylabel='intervene',
       title='Interventions over time')
ax[1].grid()


plt.show()
#
#
#
