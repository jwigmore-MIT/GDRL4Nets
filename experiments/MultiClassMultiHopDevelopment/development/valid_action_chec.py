

"""
Train on a single environment using PPO algorithm
"""


import os

import torch

from modules.torchrl_development.envs.custom_transforms import MCMHPygLinkGraphTransform, SymLogTransform
from torchrl.envs.transforms import TransformedEnv
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from modules.torchrl_development.utils.gnn_utils import tensors_to_batch
import json
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.envs.env_creation import EnvGenerator
from torchrl.envs.utils import check_env_specs
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
import wandb
import tqdm
import time
import sys
from copy import deepcopy
from torchrl.objectives.value.functional import generalized_advantage_estimate
from tensordict import TensorDict


from experiments.MultiClassMultiHopDevelopment.development.backpressure import BackpressureActor
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.objectives.value import GAE


from modules.torchrl_development.utils.metrics import compute_lta
from torchrl.modules import ProbabilisticActor, ActorCriticWrapper
import pickle




SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_PATH = os.path.dirname(SCRIPT_PATH)

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
PARAMETERS
"""
cfg = load_config(os.path.join(EXPERIMENT_PATH, 'config', 'MCMH_GNN_PPO_settings.yaml'))

# cfg.collector.total_frames = int(cfg.collector.frames_per_batch* 10)

"""
Get Environment
"""
env_name= "env2"

file_path = f"../envs/{env_name}.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

env_generator = EnvGenerator(input_params=env_info,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             mcmh=True)

eval_env_generator = EnvGenerator(input_params=env_info,
                                    make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                                    env_generator_seed = 1,
                                    mcmh=True)

env = env_generator.sample()
check_env_specs(env)


"""
CREATE GNN ACTOR CRITIC 
"""
from experiments.MultiClassMultiHopDevelopment.agents.mcmh_link_sage import MCHCGraphSage, GNN_Actor, GNN_Critic

gnn_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=cfg.agent.hidden_channels,
                            num_layers=cfg.agent.num_layers,
                            normalize=False,
                            activate_last_layer=False
                            )

actor = GNN_Actor(module = gnn_module,
                         feature_key="X", edge_index_key="edge_index",
                                     class_edge_index_key="class_edge_index", out_keys=["logits", "probs"],
                                     )

gnn_critic_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=cfg.agent.hidden_channels,
                            num_layers=cfg.agent.num_layers,
                            normalize=False,
                            activate_last_layer=False
                            )

critic = GNN_Critic(module = gnn_critic_module,
                            feature_key="X", edge_index_key="edge_index",
                            class_edge_index_key="class_edge_index", out_keys=["state_value"],
                        )


from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical
from torchrl.modules import ActorCriticWrapper
actor = ProbabilisticActor(actor,
                           in_keys = ["logits", "mask"],
                           distribution_class= MaskedOneHotCategorical,
                           spec = env.action_spec,
                           default_interaction_type = ExplorationType.RANDOM,
                           return_log_prob=True,
                            )

agent = ActorCriticWrapper(actor, critic)


td = env.rollout(max_steps=1000, policy=agent)