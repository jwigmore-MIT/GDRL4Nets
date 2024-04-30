# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
import yaml
import wandb
from train_dqn_agent import train_dqn_agent
from torchrl.record.loggers import get_logger, generate_exp_name
from argparse import ArgumentParser

""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%

parser = ArgumentParser(description='Run Agent from Sweep')
parser.add_argument('--sweep_id', type=str, help='id of the sweep to run', default="ctnere09")
args = parser.parse_args()




SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(SCRIPT_PATH, 'DQN_sweep_settings.yaml'), "r") as file:
    sweep_configuration = yaml.safe_load(file)
sweep_id = args.sweep_id
env_id = "SH1E"


def main():
    experiment_name = generate_exp_name("DQN", "Sweep1")
    logger = get_logger("wandb",
            experiment_name= experiment_name,
            logger_name="..\\logs",
            sweep_id = sweep_id,)
    run = logger.experiment
    # load base configuration
    cfg = load_config(os.path.join(SCRIPT_PATH,"DQN_settings.yaml"))
    # Modify configuration based on sweep configuration
    for top_level_key, top_level_value in run.config.items():
        for key, value in top_level_value.items():
            cfg.__dict__[top_level_key].__dict__[key] = value

    base_env_params = parse_env_json(f"{env_id}.json")
    train_dqn_agent(cfg, base_env_params, logger, disable_pbar=True)




wandb.agent(sweep_id = sweep_id, function=main, project = "Experiment15b")
