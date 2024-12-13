# test_MCMH_simulation.py
import pytest
import json
import torch
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from tensordict import TensorDict

@pytest.fixture
def env_info():
    with open("../envs/env2.json", 'r') as file:
        return json.load(file)



def test_simulation_steps(env_info):
    net = MultiClassMultiHop(**env_info)
    total_arrivals = 0
    total_departures = 0
    for t in range(1000):
        td = TensorDict({"action": net._get_random_valid_action(), "valid_action": torch.Tensor([True])})
        out = net._step(td)
        total_arrivals += out["arrivals"].sum().item()
        total_departures += out["departures"].sum().item()
        flag = total_arrivals == out["Q"].sum().item() + total_departures
        assert flag

if __name__ == "__main__":
    pytest.main()