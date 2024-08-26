import pytest
import os
import sys

from modules.torchrl_development.envs.env_creation import parse_env_json, make_env, EnvGenerator


""""
Want to test all functions in the env_creation module
This includes:
1. parse_env_json
2. make_env
3. EnvGenerator
"""

ENV_SETTINGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "env_settings")
SINGLE_ENV_DIR = os.path.join(ENV_SETTINGS_DIR, "single_envs")
CONTEXT_SETS_DIR = os.path.join(ENV_SETTINGS_DIR, "context_sets")

def test_parse_env_json():
    # Test the parse_env_json function
    env_file_name = "SH1_Poisson.json"
    env_file_path = os.path.join(SINGLE_ENV_DIR, env_file_name)
    env_dict = parse_env_json(full_path = env_file_path)
    assert env_dict["name"] == "SingleHop1"


def test_make_env():
    # Test the make_env function
    env_file_name = "SH1_Poisson.json"
    env_file_path = os.path.join(SINGLE_ENV_DIR, env_file_name)
    env_dict = parse_env_json(full_path = env_file_path)
    env = make_env(env_dict)
    assert env.name == "SingleHop1"



if __name__ == "__main__":
    test_parse_env_json()
    print("All tests passed!")

