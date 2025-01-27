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
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import multiprocessing
from torch_geometric.data import Data, Batch
from torchrl.envs import ParallelEnv


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

output = actor(network_specs)

ns2 = network_specs.clone()
state = torch.normal(0,1, size = ns2["X"].shape)
ns2["X"] = state

output = actor(ns2)

mean_location_logits = output["logits"][:,:,0].mean()
mean_scale_logits = output["logits"][:,:,1].mean()

mean_loc = output["loc"].mean()
mean_scale = output["scale"].mean()

print(f"Mean Location Logits: {mean_location_logits}"
      f"\nMean Scale Logits: {mean_scale_logits}"
      f"\nMean Location: {mean_loc}"
      f"\nMean Scale: {mean_scale}")