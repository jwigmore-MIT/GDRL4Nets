import json
import os
from torchrl_development.utils.configuration import load_config
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.actors import create_actor_critic
import torch
from torchrl.record.loggers import get_logger
from datetime import datetime
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm
from torchrl_development.utils.metrics import compute_lta
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Run experiment')
parser.add_argument('--model', type=str, help='Model to Test', default="model82a")

args = parser.parse_args()

model = args.model


SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_PATH, "experiment8_model_test.yaml")

experiment_name = "experiment82_model_test"

# Load all testing contexts
#test_context_set = json.load(open('SH2u_context_set_100_03211523.json'))
test_context_set = json.load(open('SH3_context_set_100_03251626.json'))
cfg =load_config(full_path= CONFIG_PATH)
cfg.exp_name = f"{experiment_name}-{datetime.now().strftime('%y_%m_%d-%H_%M_%S')}"
# Create a generator from test_context_set
make_env_parameters = {"observe_lambda": cfg.agent.observe_lambda,
                        "device": cfg.device,
                        "terminal_backlog": cfg.eval_envs.terminal_backlog,
                        "inverse_reward": cfg.eval_envs.inverse_reward,
                        "stat_window_size": 100000,
                        "terminate_on_convergence": True,
                        "convergence_threshold": 0.1,
                        "terminate_on_lta_threshold": True,}

env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=cfg.eval_envs.env_generator_seed)

base_env = env_generator.sample(0)
env_generator.clear_history()

input_shape = base_env.observation_spec["observation"].shape
output_shape = base_env.action_spec.space.n

# Create agent
agent = create_actor_critic(
    input_shape,
    output_shape,
    in_keys=["observation"],
    action_spec=base_env.action_spec,
    temperature=cfg.agent.temperature,
    )

# Set device
device = cfg.device

# Load agent
agent.load_state_dict(torch.load(f'{model}.pt', map_location=device))

# Set agent to eval mode
agent.eval()

# Start Wandb instance
logger = get_logger(
"wandb",
logger_name="..\\logs",
experiment_name=getattr(cfg, "exp_name", None),
wandb_kwargs={
    "config": cfg.as_dict(),
    "project": cfg.logger.project_name,
},
)
# create progress bar
pbar = tqdm(range(env_generator.num_envs), desc="Testing", unit="envs")
# Test the agent
with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

    for i in pbar:
        final_lta_backlogs = []
        log_info = {}
        for j in range(cfg.eval.num_eval_envs):
            env = env_generator.sample(i) # sample the ith environment
            td = env.rollout(cfg.eval.traj_steps, agent)
            backlogs = td["next", "backlog"].numpy()
            lta_backlog = compute_lta(backlogs)
            final_lta_backlogs.append(lta_backlog[-1])
        agent_lta = np.mean(final_lta_backlogs)
        max_weight_lta = env_generator.context_dicts[i]["lta"]
        normalize_lta = agent_lta / max_weight_lta
        log_info.update({"env_id": i, "lta": agent_lta, "normalize_lta": normalize_lta})

        pbar.set_postfix({"env_id": i, "lta": agent_lta, "normalize_lta": normalize_lta})
        for key, value in log_info.items():
            logger.log_scalar(key, value, i)
