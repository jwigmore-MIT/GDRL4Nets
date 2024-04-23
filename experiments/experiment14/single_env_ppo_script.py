from train_agent import train_ppo_agent
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl_development.utils.configuration import smart_type
import argparse
import os
from datetime import datetime
from torchrl_development.envs.env_generators import parse_env_json
import torch

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Arguments from Commmand Line
parser = argparse.ArgumentParser(description='Run experiment')
# parser.add_argument('--training_set', type=str, help='indices of the environments to train on', default="a")
parser.add_argument('--agent_type', type=str, help='type of agent to train', default="MLP")
parser.add_argument('--context_set', type=str, help='reference_to_context_set', default="SH1D") # or SH2u
parser.add_argument('--experiment_name', type=str, help='what the experiment will be titled for wandb', default="Experiment14")
parser.add_argument('--cfg', nargs = '+', action='append', type = smart_type, help = 'Modify the cfg object')

args = parser.parse_args()

# Load Config
if args.agent_type == "MLP":
    CONFIG_PATH = os.path.join(SCRIPT_PATH, "PPO_MLP_training_params.yaml")
elif args.agent_type == "MWN":
    CONFIG_PATH = os.path.join(SCRIPT_PATH, "PPO_MWN_training_params.yaml")
cfg = load_config(full_path= CONFIG_PATH)

if args.cfg:
    for key_value in args.cfg:
        keys, value = key_value
        keys = keys.split('.')
        target = cfg
        for key in keys[:-1]:
            target = getattr(target, key)
        setattr(target, keys[-1], value)

cfg.agent_type = args.agent_type

cfg.exp_name = f"{args.experiment_name}-{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}"
cfg.device = device
#print out the cfg object
print("="*20)
print(f"Experiment {cfg.exp_name}")
print("-"*20)
print("CONFIG PARAMETERS")
print("-"*20)
for key, value in cfg.as_dict().items():
    print(f"{key}: {value}")
print("="*20)



# Make Environment Generators

## Load Environment Params
cfg.context_set = args.context_set

env_params = parse_env_json(f"{cfg.context_set}.json")

training_make_env_parameters = {"observe_lambda": False,
                   "device": cfg.device,
                   "terminal_backlog": cfg.training_env.terminal_backlog,
                   "inverse_reward": cfg.training_env.inverse_reward,
                   }

training_env_generator = EnvGenerator(env_params,
                                training_make_env_parameters,
                                env_generator_seed=cfg.training_env.env_generator_seed)

# creating eval env_generators
eval_make_env_parameters = {"observe_lambda": False,
                        "device": cfg.device,
                        "terminal_backlog": cfg.eval_envs.terminal_backlog,
                        "inverse_reward": cfg.eval_envs.inverse_reward,
                        "stat_window_size": 100000,
                        "terminate_on_convergence": False,
                        "convergence_threshold": 0.01,
                        "terminate_on_lta_threshold": False,}

eval_env_generator = EnvGenerator(env_params,
                                eval_make_env_parameters,
                                env_generator_seed=cfg.eval_envs.env_generator_seed)

# Create Agent
base_env = training_env_generator.sample()
if args.agent_type == "MLP":
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=cfg.agent.temperature,
        actor_depth=cfg.agent.actor_depth,
        actor_cells=cfg.agent.actor_hidden_size
    )
elif args.agent_type == "MWN":
    agent = create_maxweight_actor_critic(
        input_shape=base_env.observation_spec["observation"].shape,
        in_keys=["Q", "Y"],
        action_spec=base_env.action_spec,
        temperature=cfg.agent.temperature,
        init_weights=torch.ones([1,2])*cfg.agent.init_weight
    )

trained_agent = train_ppo_agent(agent, training_env_generator, eval_env_generator, cfg, device =device)