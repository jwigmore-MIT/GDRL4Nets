import tensordict

from NetworkRunner import NetworkRunner
from modules.torchrl_development.envs.MultiClassMultihopBP import MultiClassMultiHopBP
import json
from experiments.GNNBiasedBackpressureDevelopment.models.node_attention_gnn import DeeperNodeAttentionGNN
from experiments.GNNBiasedBackpressureDevelopment.modules.modules import NormalWrapper
from torchrl.modules import ProbabilisticActor, IndependentNormal, TanhNormal
from tensordict.nn import InteractionType
# import optimizer
import torch.optim as optim
import tensordict
import torch
from torchrl.envs import ExplorationType, set_exploration_type
from torch.utils.tensorboard import SummaryWriter
import time



import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if __name__ == "__main__":

    file_path = "../envs/grid_3x3.json"
    env_info = json.load(open(file_path, 'r'))
    env_info["action_func"] = "bpi"
    env = MultiClassMultiHopBP(**env_info)
    network_specs = env.get_rep()

    # Create the Model
    model = DeeperNodeAttentionGNN(
        node_channels = network_specs["X"].shape[-1],
        edge_channels = network_specs["edge_attr"].shape[-1],
        hidden_channels =8,
        num_layers = 4,
        output_channels=2,
        output_activation=None,
        edge_decoder=True
    )

    norm_module = NormalWrapper(model)
    actor = ProbabilisticActor(norm_module,
                                in_keys = ["loc", "scale"],
                                out_keys = ["bias"],
                                distribution_class=IndependentNormal,
                               distribution_kwargs={"tanh_loc": True},
                                return_log_prob=True,
                                default_interaction_type = InteractionType.RANDOM
                                )
    optimizer = optim.Adam(actor.parameters(), lr=1e-10)

    output = actor(network_specs)
    runner = NetworkRunner(env = env, max_steps = 2000, actor = actor)

    from torchrl.collectors import SyncDataCollector, MultiSyncDataCollector
    create_env_func= lambda: NetworkRunner(env = env, max_steps = 2000)
    start_time = time.time()
    collector = SyncDataCollector(
        create_env_fn=create_env_func,
        policy=actor,
        frames_per_batch=4,
        total_frames=4)
    end_time = time.time()
    print(f"Time to create collector: {end_time-start_time:.2f} seconds")
    start_time = time.time()
    multicollector = MultiSyncDataCollector(
        create_env_fn=[create_env_func, create_env_func, create_env_func, create_env_func],
        policy=actor,
        frames_per_batch=4,
        total_frames=4,
        cat_results= 0
    )
    end_time = time.time()
    print(f"Time to create multi collector: {end_time-start_time:.2f} seconds")
    start_time = time.time()
    for i, td in enumerate(collector):
        print(f"Step {i}")
    end_time = time.time()
    print(f"Time for single collector: {end_time-start_time:.2f} seconds")
    start_time = time.time()

    for i, td in enumerate(multicollector):
        print(f"Step {i}")

    end_time = time.time()
    print(f"Time for multi collector: {end_time-start_time:.2f} seconds")
    # return td

    #TODO convert td to data graph

