# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
from train_dqn_agent import train_dqn_agent


""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = load_config(os.path.join(SCRIPT_PATH, "DQN_settings.yaml"))
base_env_params = parse_env_json(f"SH1E.json")

cfg.agent.num_cells = 256
cfg.optim.lr = 0.02297

train_dqn_agent(cfg, base_env_params, logger = None, disable_pbar=False)





