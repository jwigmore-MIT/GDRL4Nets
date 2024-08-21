# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
from train_dqn_agent import train_dqn_agent
from torchrl_development.envs.env_generators import make_env
from argparse import ArgumentParser

parser = ArgumentParser(description='Settings')
parser.add_argument('--device', type=str, help='id of the sweep to run', default=None)
args = parser.parse_args()



""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
if args.device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = args.device
print(f"Device: {device}")

cfg = load_config(os.path.join(SCRIPT_PATH, "DQN_settings.yaml"))
cfg.device = device
env_params = parse_env_json(f"SH1E.json")

base_env = make_env(env_params)

train_dqn_agent(cfg, env_params, device, logger = None, disable_pbar=False)









