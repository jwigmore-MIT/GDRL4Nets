import pytest
import torch
import json
import networkx as nx
from tensordict import TensorDict
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.envs.custom_transforms import MCMHPygTransform
from torchrl.envs.transforms import TransformedEnv

@pytest.fixture
def env_info():
    with open("../envs/env2.json", 'r') as file:
        return json.load(file)

@pytest.fixture
def setup_env(env_info):
    net = MultiClassMultiHop(**env_info)
    env = TransformedEnv(net, MCMHPygTransform(in_keys=["Q"], out_keys=["X"], env=net))
    return net, env

def test_edge_index_shape(setup_env):
    net, env = setup_env
    td = env.reset()
    assert td["edge_index"].shape == (2, net.M * net.K)

def test_class_edge_index_shape(setup_env):
    net, env = setup_env
    td = env.reset()
    assert td["class_edge_index"].shape == (2, net.N * net.K * (net.K - 1))

def test_X_shape(setup_env):
    net, env = setup_env
    td = env.reset()
    assert td["X"].shape == (net.N * net.K, 3)

def test_edge_index_disconnected_subgraphs(setup_env):
    net, env = setup_env
    td = env.reset()
    G2 = nx.DiGraph(td["edge_index"].T.tolist())
    assert nx.number_weakly_connected_components(G2) == net.K

def test_class_edge_index_disconnected_subgraphs(setup_env):
    net, env = setup_env
    td = env.reset()
    G3 = nx.DiGraph(td["class_edge_index"].T.tolist())
    assert nx.number_weakly_connected_components(G3) == net.N

def test_total_nodes(setup_env):
    net, env = setup_env
    td = env.reset()
    assert (td["edge_index"].unique() == td["class_edge_index"].unique()).all()