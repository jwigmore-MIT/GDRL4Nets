import random
import json
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
import time
import warnings
import torch
warnings.filterwarnings("ignore", category=DeprecationWarning)
# device should be GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"
# Count the number of cpus available
print(f"Number of CPUs: {torch.get_num_threads()}")
# Count the number of GPUs available
print(f"Number of GPUs: {torch.cuda.device_count()}")


print(f"Device: {device}")
file_path = "../envs/grid_5x5.json"
env_info = json.load(open(file_path, 'r'))
net = MultiClassMultiHop(**env_info)
net.to(device)
bp_actor = BackpressureActor(net).to(device)
start_time = time.time()
td = net.rollout(max_steps = 500, policy=bp_actor)
end_time = time.time()
rollout_time = end_time-start_time


print(f"Rollout Time: {rollout_time:.2f} seconds")