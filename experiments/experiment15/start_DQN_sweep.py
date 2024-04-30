# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
import yaml
import wandb
from train_dqn_agent import train_dqn_agent
from torchrl.record.loggers import get_logger, generate_exp_name


""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%





SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_PATH, 'DQN_sweep_settings.yaml'), "r") as file:
    sweep_configuration = yaml.safe_load(file)
sweep_id = wandb.sweep(sweep_configuration, project="Experiment15c")

