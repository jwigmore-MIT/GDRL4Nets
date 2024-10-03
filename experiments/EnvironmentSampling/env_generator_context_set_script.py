from helpers import create_env_dict, create_context_set_from_folder
from modules.torchrl_development.envs.env_creation import EnvGenerator
import json
from modules.torchrl_development.utils.configuration import make_serializable


context_set = create_context_set_from_folder("grid_environments")
make_env_keywords = {
    "observation_keys": ["q", "s", "node_priority"],
    "max_queue_size": 200,
    "stack_observation": True
}


env_generator = EnvGenerator(input_params=context_set,
                             make_env_keywords = make_env_keywords,
                             env_generator_seed = 0,
                             cgs = True)

env = env_generator.sample()

with open("../ConflictGraphSchedulingDevelopment/context_sets/grid_context_set.json", "w") as f:
    json.dump(make_serializable(context_set), f)
