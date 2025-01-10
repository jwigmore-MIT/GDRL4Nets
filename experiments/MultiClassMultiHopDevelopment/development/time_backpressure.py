import random
import json
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform
from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor, BackpressureGNN_Actor
import time
import warnings
import torch
from torchrl.envs import ParallelEnv
from torchrl.envs.transforms import TransformedEnv
from torchrl.envs import ExplorationType
from torchrl.modules import ProbabilisticActor
from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical



warnings.filterwarnings("ignore", category=DeprecationWarning)
# device should be GPU if available

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    # Count the number of cpus available
    num_cpus = torch.get_num_threads()
    print(f"Number of CPUs: {num_cpus}")
    # Count the number of GPUs available
    print(f"Number of GPUs: {torch.cuda.device_count()}")


    print(f"Device: {device}")
    file_path = "../envs/grid_5x5.json"
    env_info = json.load(open(file_path, 'r'))
    # env_info["action_func"] = "bpi"

    # Using BackpressureGNN_Actor
    """
    net = MultiClassMultiHop(**env_info)
    net = TransformedEnv(net, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=net))
    bp_actor = BackpressureGNN_Actor()
    bp_actor = ProbabilisticActor(bp_actor,
                                in_keys = ["logits", "mask"],
                                distribution_class= MaskedOneHotCategorical,
                                spec = net.action_spec,
                                default_interaction_type = ExplorationType.MODE,
                                return_log_prob=True,
                                 )
    start_time = time.time()
    td = net.rollout(max_steps = 500, policy=bp_actor)
    end_time = time.time()
    rollout_time = end_time-start_time
    
    
    print(f"Rollout Time: {rollout_time:.2f} seconds")
    """

    # Using MultiClassMultiHopBP environment
    """
    net = MultiClassMultiHopBP(**env_info)
    start_time = time.time()
    td = net.rollout(max_steps = 500)
    end_time = time.time()
    rollout_time = end_time-start_time
    print(f"Rollout Time: {rollout_time:.2f} seconds")
    """
    for i in range(5):
        # Using ParallelEnv and MultiClassMultiHopBP environment
        make_net = lambda: MultiClassMultiHopBP(**env_info)
        net = ParallelEnv(num_cpus, make_net)
        start_time = time.time()
        td = net.rollout(max_steps = 500)
        end_time = time.time()
        rollout_time = end_time-start_time
        print(f"Rollout Time: {rollout_time:.2f} seconds")
    return td

if __name__ == '__main__':
    td = main()