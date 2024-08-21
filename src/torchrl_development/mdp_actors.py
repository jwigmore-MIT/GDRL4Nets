import torch
import numpy as np
from tensordict.nn import TensorDictModule
from tensordict import TensorDict
import json
import pickle
from torchrl_development.envs.env_generators import make_env, parse_env_json
from torchrl_development.utils.metrics import compute_lta
import matplotlib.pyplot as plt
class MDP_actor(TensorDictModule):

    def __init__(self, mdp_module, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= mdp_module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["Y"])
        return td

class MDP_module(torch.nn.Module):

    def __init__(self, mdp):
        super().__init__()
        self.mdp = mdp

    def forward(self, Q, Y):
        state = torch.concatenate([Q, Y]).tolist()
        return self.mdp.use_policy(state)