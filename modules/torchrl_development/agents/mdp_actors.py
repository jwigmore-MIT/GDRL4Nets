import torch
from tensordict import TensorDict
from tensordict.nn import (
    TensorDictModule,)

"""
For taking MDP objects and enabling them to interact in TorchRL environments
"""

class MDP_actor(TensorDictModule):

    def __init__(self, mdp_module, in_keys = ["Q", "Y"], out_keys = ["action"]):
        super().__init__(module= mdp_module, in_keys = in_keys, out_keys=out_keys)

    def forward(self, td: TensorDict):
        td["action"] = self.module(td["Q"], td["Y"])
        return td

class MDP_module(torch.nn.Module):

    def __init__(self, mdp, policy_type = "VI"):
        super().__init__()
        self.mdp = mdp
        self.policy_type = policy_type


    def forward(self, Q, Y):
        state = torch.concatenate([Q, Y]).tolist()
        if self.policy_type == "VI":
            return self.mdp.use_vi_policy(state)
        elif self.policy_type == "PI":
            return self.mdp.use_pi_policy(state)
        else:
            raise ValueError(f"Unknown policy type {self.policy_type}")


