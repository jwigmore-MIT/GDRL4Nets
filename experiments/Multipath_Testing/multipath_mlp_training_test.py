from torchrl_development.envs.env_generators import make_env, parse_env_json
import tensordict as td
import matplotlib.pyplot as plt
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
import os
from torchrl_development.utils.configuration import load_config


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))



env_params = parse_env_json("MP1.json")
env= make_env(env_params,
                observation_keys=["arrivals", "Q", "Y"])

input_shape = env.observation_spec["observation"].shape
output_shape = env.action_spec.shape
action_spec = env.action_spec

agent = create_actor_critic(input_shape,
                            output_shape,
                            in_keys=["observation"],
                            action_spec=action_spec,
                            temperature=1,
                            actor_depth = 2,
                            actor_cells= 64
                            )

