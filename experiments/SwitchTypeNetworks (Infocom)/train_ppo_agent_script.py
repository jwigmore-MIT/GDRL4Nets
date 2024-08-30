# Import the necessary libraries
import os
import torch
import json
import argparse

# Import the necessary functions from the torchrl library
from torchrl.record.loggers import get_logger, generate_exp_name

# Import the necessary functions from the torchrl_development library
from modules.torchrl_development.utils.configuration import load_config, smart_type
from modules.torchrl_development.envs.env_creation import EnvGenerator

# Import necessary function from experiment's training script
from train_ppo_agent import train_ppo_agent


""" Explanation of Script
The following script is used to train a single agent on one or more environments. 
These environments are specified by selecting a context set, which contains many different environments,
and then selecting a range of integers that correspond to the environments in the context set that you want to train on.
The script will then train the agent on the specified environments and log the results to wandb.
The script is run from the command line and takes in the following arguments:
    --training_code: a tuple of integers that specify the range of environments to train on
    --context_set: the name of the context set to use
    --train_type: the base configuration file to use
    --cfg: a list of modifications to the configuration file

"""


# Prevents crashing when uploading to wandb
os.environ["MAX_IDLE_COUNT"] = "1_000_000"

# Get script path
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
# LOGGING PATH IS PARENT OF PARENT OF SCRIPT PATH
LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")
""" This will run a single experimemt of the monotonic DQN agent on the specified context set
"""
# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--training_code', type = tuple, help='range of integers to train on', default=(0,5))
    parser.add_argument('--testing_code', type = tuple, help='range of integers to test on', default=None)
    parser.add_argument('--train_type', type=str, help='base configuration file', default="MLP_PPO")
    parser.add_argument('--cfg', nargs = '+', action='append', type = smart_type, help = 'Modify the cfg object')

    base_cfg = {
                "MLP_PPO": 'MLP_PPO_settings.yaml',  # For Single-hop
                "MLP_PPO_MP": 'MLP_PPO_MP_settings.yaml', # For Multi-path
                "STN_shared_PPO": 'STN_Shared_PPO_settings.yaml',
                "STN_shared_PPO_MP": 'STN_Shared_PPO_MP_settings.yaml',}


    # Parse the arguments
    args = parser.parse_args()

        # Specify which environments to train on from the context set by providing a range of integers
    training_env_ind = range(args.training_code[0], args.training_code[0] + args.training_code[1])

    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Device: {device}")

    # Load configuration file for the training
    cfg = load_config(os.path.join(SCRIPT_PATH, 'config', base_cfg[args.train_type]))
    cfg.device = device


    context_set_json = "MP5_context_set_l3_m1_s100.json" if "MP" in args.train_type else "SH4_context_set_l3_m3_s100.json"

    # Select the Context Set
    context_set = json.load(open(os.path.join(SCRIPT_PATH, 'context_sets', context_set_json), "r"))
    cfg.context_set = args.context_set
    all_context_dicts = context_set['context_dicts']



    if args.cfg:
        for key_value in args.cfg:
            keys, value = key_value
            keys = keys.split('.')
            target = cfg
            for key in keys[:-1]:
                target = getattr(target, key)
            setattr(target, keys[-1], value)

    # env_params = parse_env_json(f"{args.env_params}.json")

    # Create the Training Env Generators
    training_make_env_parameters = {"graph": getattr(cfg.training_env, "graph", False),
                                    "observe_lambda": getattr(cfg.training_env, "observe_lambda", True),
                                    "observe_mu": getattr(cfg.training_env, "observe_mu", True),
                                    "terminal_backlog": getattr(cfg.training_env, "terminal_backlog", None),
                                    "observation_keys": getattr(cfg.training_env, "observation_keys", ["Q", "Y"]),
                                    "observation_keys_scale": getattr(cfg.training_env, "observation_keys_scale", None),
                                    "negative_keys": getattr(cfg.training_env, "negative_keys", ["mu"]),
                                    "symlog_obs": getattr(cfg.training_env, "symlog_obs", False),
                                    "symlog_reward": getattr(cfg.training_env, "symlog_reward", False),
                                    "inverse_reward": getattr(cfg.training_env, "inverse_reward", False),
                                    "cost_based": getattr(cfg.training_env, "cost_based", True),
                                    "reward_scale": getattr(cfg.training_env, "reward_scale", 1.0),
                                    "stat_window_size": getattr(cfg.training_env, "stat_window_size", 5000),}

    # training_env_ind = train_sets[args.training_set]["train"]

    training_env_generator_input_params = {"context_dicts": {i: all_context_dicts[str(i)] for i in training_env_ind},
                                           "num_envs": len(training_env_ind), }

    training_env_generator = EnvGenerator(training_env_generator_input_params,
                                          training_make_env_parameters,
                                          env_generator_seed=cfg.training_env.env_generator_seed,
                                          cycle_sample=False)

    # Create Test Env Generators
    test_make_env_parameters = training_make_env_parameters.copy()
    test_env_ind = []
    test_env_ind.extend(training_env_ind) # adds the training indices to the test indices so all training environments, are also tested
    if args.testing_code:
        test_env_ind = range(args.testing_code[0], args.testing_code[0] + args.testing_code[1])
    test_env_generator_input_params = {"context_dicts": {i: all_context_dicts[str(i)] for i in test_env_ind},
                                       "num_envs": len(test_env_ind), }

    test_env_generator = EnvGenerator(test_env_generator_input_params,
                                      test_make_env_parameters,
                                      env_generator_seed=cfg.training_env.env_generator_seed, )

    # Create wandb Logger
    experiment_name = generate_exp_name(f"{args.train_type}", f"{cfg.context_set}_{args.training_code[0]}_{args.training_code[1]}")
    logger = get_logger(
            "wandb",
            logger_name=LOGGING_PATH,
            experiment_name= experiment_name,
            wandb_kwargs={
                "config": cfg.as_dict(),
                "project": cfg.logger.project,
            },
        )


    train_ppo_agent(cfg, training_env_generator, test_env_generator, device, logger)




