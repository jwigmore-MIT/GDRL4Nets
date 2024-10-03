import os
# add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# print(sys.executable)
# print(sys.path)
import os
from torchrl.envs.utils import check_env_specs
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.collectors import SyncDataCollector
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.record.loggers import get_logger, generate_exp_name
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.functional import generalized_advantage_estimate

from graph_env_creators import make_line_graph, make_ring_graph, create_grid_graph
from modules.torchrl_development.utils.configuration import load_config
from modules.torchrl_development.agents.cgs_agents import create_mlp_actor_critic, GNN_ActorTensorDictModule
from modules.torchrl_development.envs.ConflictGraphScheduling import ConflictGraphScheduling, compute_valid_actions
from modules.torchrl_development.baseline_policies.maxweight import CGSMaxWeightActor
from modules.torchrl_development.envs.env_creation import make_env_cgs, EnvGenerator
from modules.torchrl_development.utils.metrics import compute_lta
from torchrl.modules import ProbabilisticActor, ActorCriticWrapper
from modules.torchrl_development.agents.cgs_agents import IndependentBernoulli, GNN_TensorDictModule, tensors_to_batch
from torch_geometric.nn import global_add_pool
import pickle
from policy_modules import *
from imitation_learning_utils import *
from torch_geometric.data import DataLoader
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch
import os
import wandb
from copy import deepcopy
import time
import tqdm
import sys
from experiment_utils import evaluate_agent




import torch._dynamo
# torch._dynamo.config.suppress_errors = True
#
# TORCH_LOGS="+dynamo"
# TORCHDYNAMO_VERBOSE=1
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
CREATE ENVIRONMENT
"""
adj, arrival_dist, arrival_rate, service_dist, service_rate = make_line_graph(8, 0.4, 1)
# adj, arrival_dist, arrival_rate, service_dist, service_rate = make_ring_graph(10, 0.4, 1)
# adj, arrival_dist, arrival_rate, service_dist, service_rate = create_grid_graph(2, 2, 0.4, 1)
G = nx.from_numpy_array(adj)

# Draw the graph
nx.draw(G, with_labels=True)
plt.title(f"Testing Network graph")
plt.show()
interference_penalty = 0.25
reset_penalty = 100

env_params = {
    "adj": adj,
    "arrival_dist": arrival_dist,
    "arrival_rate": arrival_rate,
    "service_dist": service_dist,
    "service_rate": service_rate,
    "env_type": "CGS",
    "interference_penalty": interference_penalty,
    "reset_penalty": reset_penalty,
    "node_priority": "increasing",

}

cfg = load_config(os.path.join(SCRIPT_PATH, 'config', 'CGS_GNN_PPO_settings.yaml'))
# cfg.training_make_env_kwargs.observation_keys = ["q"]
# cfg.training_make_env_kwargs.observation_keys.append("node_priority") # required to differentiate between nodes with the same output embedding

cfg.agent.hidden_size = 2
cfg.agent.num_layers = 8



env_generator = EnvGenerator(input_params=env_params,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             cgs = True)

eval_env_generator = EnvGenerator(input_params=env_params,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             cgs = True)


env = env_generator.sample()

check_env_specs(env)

"""
CREATE GNN ACTOR CRITIC
"""
node_features = env.observation_spec["observation"].shape[-1]
policy_module = Policy_Module2(node_features, cfg.agent.hidden_size, num_layers = cfg.agent.num_layers, dropout=0.1)
# policy_module = GCN_Policy_Module(node_features, num_layers = cfg.agent.num_layers)
# policy_module = torch.compile(policy_module)

actor = GNN_ActorTensorDictModule(module = policy_module, x_key = "observation", edge_index_key = "adj_sparse", out_keys = ["probs", "logits"])


# value_module = Value_Module(node_features, cfg.agent.hidden_size, num_layers = cfg.agent.num_layers, dropout = 0.1)

# critic = GNN_TensorDictModule(module = value_module, x_key="observation", edge_index_key="adj_sparse", out_key="state_value")

actor = ProbabilisticActor(
    actor,
    in_keys=["probs"],
    distribution_class=IndependentBernoulli,
    spec = env.action_spec,
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM
    )

# agent = ActorCriticWrapper(actor, critic)
agent = actor

def collect_rollouts(max_steps = 50):
    times = []
    td = env.rollout(max_steps=max_steps, break_when_all_done=False, break_when_any_done=False)

    for n in range(5):
        start_time = time.time()
        td = env.rollout(max_steps=max_steps, policy = agent, break_when_any_done=False, break_when_all_done=False)
        # td = env.rollout(max_steps=5000, break_when_all_done=False, break_when_any_done=False)
        end_time = time.time()
        print(f"Time taken for rollout: {end_time - start_time}")
        times.append(end_time - start_time)
    print(f"Average time taken for rollout: {np.mean(times)}")

def clock_forward_pass(max_steps = 50):
    times = []
    td = env.rollout(max_steps=max_steps, break_when_all_done=False, break_when_any_done=False)
    for n in range(5):
        start_time = time.time()
        td = agent(td)
        end_time = time.time()
        print(f"Time taken for forward pass: {end_time - start_time}")
        times.append(end_time - start_time)
    print(f"Average time taken for forward pass: {np.mean(times)}")

def clock_repeat_forward_pass(repeat = 50):
    times = []
    td = env.rollout(max_steps=1, break_when_all_done=False, break_when_any_done=False)
    for n in range(5):
        start_time = time.time()
        for i in range(repeat):
            td = agent(td)
        end_time = time.time()
        print(f"Time taken for forward pass: {end_time - start_time}")
        times.append(end_time - start_time)
    print(f"Average time taken for forward pass: {np.mean(times)}")



collect_rollouts(50)
clock_forward_pass(50)
clock_repeat_forward_pass(50)

