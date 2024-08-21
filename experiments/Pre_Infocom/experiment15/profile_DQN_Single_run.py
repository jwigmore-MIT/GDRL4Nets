# %%
from torchrl_development.utils.configuration import load_config
import os
from torchrl_development.envs.env_generators import parse_env_json
import torch
from train_dqn_agent import train_dqn_agent
import cProfile
import pstats


""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
cfg = load_config(os.path.join(SCRIPT_PATH, "DQN_settings.yaml"))
base_env_params = parse_env_json(f"SH1E.json")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg.device = device

# cfg.agent.num_cells = 128
# cfg.buffer.batch_size = 20
# cfg.collector.frames_per_batch = 10
# cfg.loss.mask_loss = 0
# cfg.loss.soft_eps = 0.95
# cfg.optim.lr = 0.0005
# cfg.agent.mask = True
cfg.collector.total_frames = 20000
# Create a profiler object
profiler = cProfile.Profile()

# Start the profiler
profiler.enable()

# Run the function you want to profile
train_dqn_agent(cfg, base_env_params, device, logger=None, disable_pbar=False)

# Stop the profiler
profiler.disable()

# Create a Stats object to format and print the data collected by the profiler
stats = pstats.Stats(profiler)

# Sort the statistics by the cumulative time spent in the function
stats.sort_stats(pstats.SortKey.CUMULATIVE)

# Print only the top 200 lines of the statistics
stats.print_stats(200)





