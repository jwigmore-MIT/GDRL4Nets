
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


if __name__ == "__main__":
    # Get context dict
    context_set_dict = json.load(open("SH1_context_set.json", 'rb'))
    context_dict = context_set_dict["context_dicts"]["0"]
    # Get MDP
    mdp1 = pickle.load(open("saved_mdps/SH1_0_MDP.p", 'rb'))
    mdp_module = MDP_module(mdp1)
    # Create actor
    mdp_actor = MDP_actor(mdp_module)

    make_env_keywords = {
        "stat_window_size": 100000,
        "terminate_on_convergence": True,
        "convergence_threshold": 0.05
    }
    env = make_env(context_dict["env_params"], **make_env_keywords)
    # Test actor
    td = env.rollout(policy=mdp_actor, max_steps = 10000)
    lta = compute_lta(td["backlog"])

    fig, ax = plt.subplots(1,1, figsize = (15,10))
    ax.plot(lta, label = "Optimal Policy")
    # plot horizontal line at the lta of in context_dict["lta"]
    ax.axhline(y=context_dict["lta"], color='r', linestyle='-', label = "MaxWeight Policy")
    # ax.set_title(f"Backlog LTA for {context_dict['env_params']['arrival_rates']}")
    ax.legend()
    fig.show()


    # now plot the lta normalized by the max_weight lta backlog in context_dict["lta"]

    fig, axes = plt.subplots(1,1, figsize = (15,10))
    axes.plot(lta/context_dict["lta"], label = "Optimal Policy")
    # plot horizontal line at the lta of in context_dict["lta"]
    axes.axhline(y=1, color='r', linestyle='-', label = "MaxWeight Policy")
    # axes.set_title(f"Normalized Backlog LTA for {context_dict['env_params']['arrival_rates']}")
    axes.legend()
    fig.show()








