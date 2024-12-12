# test_torchrl.py
import pytest
import json
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from torchrl.envs import check_env_specs

@pytest.fixture
def env_info():
    with open("../envs/env2.json", 'r') as file:
        return json.load(file)

def test_env_creation(env_info):
    net = MultiClassMultiHop(**env_info)
    assert net is not None

def test_check_env_specs(env_info):
    net = MultiClassMultiHop(**env_info)
    check_env_specs(net)