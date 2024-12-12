# test_backpressure.py
import json
import torch
import pytest
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from experiments.MultiClassMultiHopDevelopment.development.backpressure import BackpressureActor


@pytest.fixture
def env_info():
    with open("../envs/env2.json", 'r') as file:
        return json.load(file)


@pytest.fixture
def net(env_info):
    return MultiClassMultiHop(**env_info)


@pytest.fixture
def bp_actor(net):
    return BackpressureActor(net)


def test_backpressure_initialization(bp_actor):
    assert bp_actor.M == 6
    assert bp_actor.K == 4
    assert len(bp_actor.link_info) == 6


def test_backpressure_algorithm(bp_actor, net):
    td = net.reset()
    T = 1000
    total_arrivals = td["arrivals"].sum().item()
    total_departures = 0
    backlog = torch.zeros([T])

    for t in range(T):
        if t < 1:
            td = bp_actor(td)
        else:
            td = bp_actor(td["next"])
        td = net.step(td)
        backlog[t] = td["next", "Q"].sum()
        total_departures += td["next", "departures"].sum().item()
        total_arrivals += td["next", "arrivals"].sum().item()
        assert total_arrivals == td["next", "Q"].sum().item() + total_departures

    assert backlog.sum().item() > 0




if __name__ == "__main__":
    pytest.main([__file__])