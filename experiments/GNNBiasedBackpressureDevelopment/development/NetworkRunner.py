from copy import deepcopy
from typing import Optional, Union, List, Dict
from collections import OrderedDict
import networkx as nx

import numpy as np
import torch
from tensordict import TensorDict, merge_tensordicts

from torchrl.data import Composite, Bounded, Unbounded, Binary, NonTensorSpec, NonTensor
from tensordict import NonTensorData

from torchrl.envs import (
    EnvBase,
)

import multiprocessing
from torch_geometric.data import Data

# ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
Essentially going to take a BPMultiClassMultiHop environment, and instead of returning each step,
- input: network, bias
- output: average backlog 

Initialized with a network, number of rollout steps

"""

def create_network_runner(env: EnvBase, max_steps: int = 1000, actor = None, graph = False, link_rewards = "shared"):
    if graph:
        return NetworkRunnerGraph(env = env, max_steps = max_steps, actor = actor, link_rewards=link_rewards)
    else:
        return NetworkRunnerTensor(env = env, max_steps = max_steps, actor = actor, link_rewards=link_rewards)


class NetworkRunnerTensor(EnvBase): # Modeled after torhrl EnvBase
    def __init__(self, env: EnvBase, max_steps: int = 1000, actor = None, **kwargs):
        super().__init__()
        self.env = env
        self.max_steps = max_steps
        self.actor = actor
        self.create_spec()

    def create_spec(self):
        env_rep = self.env.get_rep()
        self.observation_spec = Composite({
        })
        for key in env_rep.keys():
            self.observation_spec[key] = Unbounded(shape = env_rep[key].shape)

        self.action_spec = Composite({
            "bias": Unbounded(shape =self.env.bias.shape)
        })


    def _reset(self, *args, **kwargs):
        return self.env.get_rep()

    def _set_seed(self, seed: Optional[int] = None):
        return
    def _step(self, td: TensorDict = None) -> TensorDict:
        if td is None:
            return self.get_run()
        else:
            return self.get_run(bias = td["bias"])
    def get_run(self, bias = None):
        td = self.env.get_rep()
        if bias is not None:
            td["bias"] = bias
            self.env.set_bias(bias)
        elif self.actor is not None:
            td = self.actor(td)
            self.env.set_bias(td["bias"].squeeze())
        else:
            td["bias"]=env.bias.clone()
        rollout = self.env.rollout(max_steps = self.max_steps)
        td["reward"] = -rollout["backlog"].mean()
        td["bias"] = td["bias"].squeeze()
        return td

    def get_runs(self, n: int = 4) -> List[TensorDict]:
        td = self.env.get_rep()
        # get n biases from the actor
        if self.actor is not None:
            td = self.actor(td)
            biases = td["bias"]

        with multiprocessing.Pool(n) as p:
            return p.map(self.get_run, [None]*n)



class NetworkRunnerGraph(EnvBase): # Modeled after torhrl EnvBase
    def __init__(self, env: EnvBase, max_steps: int = 1000, actor = None, link_rewards = "shared", **kwargs):
        super().__init__()
        self.env = env
        self.max_steps = max_steps
        self.actor = actor
        self.create_spec()
        self.link_rewards = link_rewards

    def create_spec(self):

        self.observation_spec = Composite({
            "data": NonTensor(shape = (1,), dtype = torch.float32),
            "Qavg": Unbounded(shape =-1),
        })
        self.action_spec = Composite({
            "bias": Unbounded(shape =-1)
        })


    def _reset(self, *args, **kwargs):
        return self.env.get_rep()

    def _set_seed(self, seed: Optional[int] = None):
        return
    def _step(self, td: TensorDict = None) -> TensorDict:
        if td is None:
            td = self.env.get_rep()
        return self.get_run(td)
    def get_run(self, td):
        if "data" in td:
            Warning(" `data` already in td")
        if "bias" in td:
            self.env.set_bias(td["bias"])
        else:
            td["bias"]=self.env.bias.clone()
        rollout = self.env.rollout(max_steps = self.max_steps)
        td["backlog"] = rollout["backlog"].mean()
        td["reward"] = -td["backlog"]
        td["Qavg"] = rollout["Q"].mean(dim =0)
        if self.link_rewards == "shared":
            td["link_rewards"] = td["reward"]*torch.ones_like(td["bias"])
        elif self.link_rewards == "Qdiff":
            td["link_rewards"] = (rollout["Q"].mean(dim =0)[self.env.end_nodes] - rollout["Q"].mean(dim =0)[self.env.start_nodes])
        graph = Data()
        for key in td.keys():
            if key == "data":
                pass
            else:
                try:
                    graph[key] = td[key].clone()
                except:
                    graph[key] = td[key]

        # td["bias"] = td["bias"].squeeze()
        # graph = Data(x = td["X"], edge_index = td["edge_index"], edge_attr = td["edge_attr"],
        #              bias = td["bias"].squeeze(), reward = reward)
        return TensorDict({"data": NonTensorData(graph),
                           "reward": td["reward"],
                           "backlog": td["backlog"]})








if __name__ == "__main__":
    from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
    import json
    file_path = "../envs/grid_3x3.json"
    env_info = json.load(open(file_path, 'r'))
    env_info["action_func"] = "bpi"
    env = MultiClassMultiHopBP(**env_info)
    runner = NetworkRunnerTensor(env = env, max_steps = 2000)
    td = runner.get_run()

