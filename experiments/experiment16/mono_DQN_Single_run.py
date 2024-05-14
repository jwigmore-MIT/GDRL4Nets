''# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
#from train_monotonic_dqn_agent import train_mono_dqn_agent
from train_monotonic_dqn_agent import train_mono_dqn_agent
from train_ppo_agent import train_ppo_agent
import json
import argparse
import numpy as np
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl.record.loggers import get_logger, generate_exp_name

os.environ["MAX_IDLE_COUNT"] = "1_000_000"

# Get script path
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))


""" This will run a single experimemt of the monotonic DQN agent on the specified context set
"""
# %%
def smart_type(value):

    if ',' in value:
        try:
            value_list = [float(item) for item in value.split(',')]
            return np.array(value_list)
        except ValueError:
            pass

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass


    if value.lower() in ['true', 'false']:
        return value.lower() == 'true'

    return value

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument('--training_set', type=str, help='indices of the environments to train on', default="b")
    parser.add_argument('--context_set', type=str, help='reference_to_context_set', default="SH2u") # or SH2u
    parser.add_argument('--train_type', type=str, help='base configuration file', default="MLP_PPO")
    parser.add_argument('--cfg', nargs = '+', action='append', type = smart_type, help = 'Modify the cfg object')

    base_cfg = {"PMN_DQN": 'PMN_DQN_settings.yaml',
                "MLP_PPO": 'MLP_PPO_settings.yaml',}

    context_set_jsons = {"SH3": "SH3_context_set_100_03251626.json",
                         "SH2u": "SH2u_context_set_10_03211514.json",}

    train_sets = {"a": {"train": [0], "test": [7,8,9]}, # lta backlog = 135.10
                  "b": {"train": [0,1,2], "test": [7,8,9]}, # 53.94
                  "c": {"train": [0,1,2,3,4,5], "test": [7,8,9]},} # 12.048

    args = parser.parse_args()


    # Set Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    print(f"Device: {device}")
    cfg = load_config(os.path.join(SCRIPT_PATH, base_cfg[args.train_type]))
    cfg.device = device

    # Select the Context Set
    context_set = json.load(open(os.path.join(SCRIPT_PATH, context_set_jsons[args.context_set]), "r"))
    cfg.context_set = args.context_set
    all_context_dicts = context_set['context_dicts']

    # Create the Training Env Generators
    training_make_env_parameters = {"graph": getattr(cfg.training_env, "graph", False),
                                    "observe_lambda": getattr(cfg.training_env, "observe_lambda", True),
                                    "terminal_backlog": getattr(cfg.training_env, "terminal_backlog", None),
                                    "observation_keys": getattr(cfg.training_env, "observation_keys", ["Q", "Y"]),
                                    "negative_keys": getattr(cfg.training_env, "negative_keys", ["Y"]),
                                    "symlog_obs": getattr(cfg.training_env, "symlog_obs", False),
                                    "symlog_reward": getattr(cfg.training_env, "symlog_reward", False),
                                    "inverse_reward": getattr(cfg.training_env, "inverse_reward", False),
                                    "cost_based": getattr(cfg.training_env, "cost_based", True),
                                    "reward_scale": getattr(cfg.training_env, "reward_scale", 1.0),
                                    "stat_window_size": getattr(cfg.training_env, "stat_window_size", 5000),}

    training_env_ind = train_sets[args.training_set]["train"]

    training_env_generator_input_params = {"context_dicts": {i: all_context_dicts[str(i)] for i in training_env_ind},
                                           "num_envs": len(training_env_ind),}

    training_env_generator = EnvGenerator(training_env_generator_input_params,
                                          training_make_env_parameters,
                                          env_generator_seed = cfg.training_env.env_generator_seed,
                                          cycle_sample=False)


    # Create Test Env Generators
    test_make_env_parameters = training_make_env_parameters.copy()
    test_env_ind = train_sets[args.training_set]["test"]
    test_env_ind.extend(training_env_ind)

    test_env_generator_input_params = {"context_dicts": {i: all_context_dicts[str(i)] for i in test_env_ind},
                                          "num_envs": len(test_env_ind),}

    test_env_generator = EnvGenerator(test_env_generator_input_params,
                                      test_make_env_parameters,
                                      env_generator_seed = cfg.training_env.env_generator_seed,)


    # Create Logger
    experiment_name = generate_exp_name(f"{args.train_type}", f"{cfg.context_set}_{args.training_set}")
    logger = get_logger(
            "wandb",
            logger_name="..\\logs",
            experiment_name= experiment_name,
            wandb_kwargs={
                "config": cfg.as_dict(),
                "project": cfg.logger.project,
            },
        )
    # from torchrl.envs import ParallelEnv
    # create_env_funcs = [training_env_generator.sample for i in range(training_env_generator.num_envs)]
    # parallel_envs = ParallelEnv(num_workers = training_env_generator.num_envs, create_env_fn = create_env_funcs)

    # cfg.collector.test_interval = 2000
    if args.train_type == "PMN_DQN":
        train_mono_dqn_agent(cfg, training_env_generator, test_env_generator, device, logger)
    elif args.train_type == "MLP_PPO":
        train_ppo_agent(cfg, training_env_generator, test_env_generator, device, logger)

    """ Needs
    
             
    
    """


