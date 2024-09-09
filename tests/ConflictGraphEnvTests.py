
import pytest
import torch
import numpy as np
from modules.torchrl_development.envs.ConflictGraphScheduling import ConflictGraphScheduling, compute_valid_actions
from torchrl.envs.utils import check_env_specs
from modules.torchrl_development.baseline_policies.maxweight import CGSMaxWeightActor
from modules.torchrl_development.envs.env_creation import make_env_cgs


@pytest.fixture
def env_setup():
    adj = np.array([[0,1,0], [1,0,1], [0,1,0]])
    arrival_rate = np.array([0.1, 0.2, 0.3])
    service_rate = np.array([1, 1, 1])

    return adj, arrival_rate, service_rate

@pytest.fixture
def env_setup2():
    adj = np.array([[0, 1, 1, 1, 0], [1, 0, 1, 1, 0], [1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])
    arrival_dist = "Bernoulli"
    arrival_rate = np.array([0.1, 0.2, 0.3, 0.1, 0.1])
    service_dist = "Fixed"
    service_rate = np.array([1, 1, 1, 1, 1])
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    return env

@pytest.fixture
def env_setup3():
    adj = np.array([[0, 1, 0, 0, ], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    arrival_dist = "Bernoulli"
    arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
    service_dist = "Fixed"
    service_rate = np.array([1, 1, 1, 1])
    max_queue_size = 1000
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate,
                                  max_queue_size=max_queue_size)
    return env

def test_poisson_arrival_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Poisson"
    service_dist = "Fixed"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    arrivals = env._sim_arrivals()
    assert isinstance(arrivals, torch.Tensor)

def test_bernoulli_arrival_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Bernoulli"
    service_dist = "Fixed"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    arrivals = env._sim_arrivals()
    assert isinstance(arrivals, torch.Tensor)

def test_fixed_arrival_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Fixed"
    service_dist = "Fixed"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    arrivals = env._sim_arrivals()
    assert isinstance(arrivals, torch.Tensor)

def test_poisson_service_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Fixed"
    service_dist = "Poisson"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    services = env._sim_services()
    assert isinstance(services, torch.Tensor)

def test_bernoulli_service_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Fixed"
    service_dist = "Bernoulli"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    services = env._sim_services()
    assert isinstance(services, torch.Tensor)

def test_fixed_service_process(env_setup):
    adj, arrival_rate, service_rate = env_setup
    arrival_dist = "Fixed"
    service_dist = "Fixed"
    env = ConflictGraphScheduling(adj, arrival_dist, arrival_rate, service_dist, service_rate)
    services = env._sim_services()
    assert isinstance(services, torch.Tensor)




def test_get_valid_action(env_setup2):
    env = env_setup2

    action1 = torch.Tensor([1, 0, 0, 0, 1])
    true_valid_action_1 = torch.Tensor([1, 0, 0, 0, 1])

    action2 = torch.Tensor([1, 1, 0, 0, 1])
    true_valid_action_2 = torch.Tensor([0, 0, 0, 0, 1])

    action3 = torch.Tensor([0,0,1,0,0])
    true_valid_action_3 = torch.Tensor([0,0,1,0,0])

    action4 = torch.Tensor([1,1,1,1,1])
    true_valid_action_4 = torch.Tensor([0,0,0,0,0])

    action = torch.stack([action1, action2, action3, action4], dim = 0)
    true_valid_action = torch.stack([true_valid_action_1, true_valid_action_2, true_valid_action_3, true_valid_action_4], dim = 0)
    valid_action = env._get_valid_action(action)

    assert torch.all(valid_action == true_valid_action)

    single_action = torch.Tensor([1, 0, 0, 0, 1])
    valid_single_action = env._get_valid_action(single_action)
    assert torch.all(valid_single_action == true_valid_action_1)

def test_check_env_specs(env_setup2):
    check_env_specs(env_setup2)




def test_make_env_cgs(env_setup):
    adj, arrival_rate, service_rate = env_setup
    env_params = {"adj": adj, "arrival_rate": arrival_rate, "service_rate": service_rate, "arrival_dist": "Bernoulli", "service_dist": "Fixed"}

    make_env_keywords = {"observation_keys": ["q", "s"]}
    env = make_env_cgs(env_params, **make_env_keywords)
    td = env.reset()
    # check to make sure in td["observation"] tensor, elements 0, 2, ... are the same and 1, 3, ... are the same
    assert torch.all(td["observation"][0::2] == td["observation"][0])
    assert torch.all(td["observation"][1::2] == td["observation"][1])

def test_make_env_cgs_stack(env_setup):
    adj, arrival_rate, service_rate = env_setup
    env_params = {"adj": adj, "arrival_rate": arrival_rate, "service_rate": service_rate, "arrival_dist": "Bernoulli", "service_dist": "Fixed"}

    make_env_keywords = {"observation_keys": ["q", "s"], "stack_observation": True}
    env = make_env_cgs(env_params, **make_env_keywords)
    td = env.reset()
    print()

def test_maxweight_actor(env_setup3):
    env = env_setup3
    # create cgs actor
    actor = CGSMaxWeightActor(valid_actions=compute_valid_actions(env))

    # Do 100 steps with actor
    td = env.rollout(max_steps=10000, policy=actor)

    action_means = td["action"].mean(dim=0)

    assert torch.all(torch.abs(action_means - 0.4 * torch.ones_like(action_means)) < 0.25)






