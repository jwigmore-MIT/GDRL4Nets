# import

import os
from torchrl.envs.utils import check_env_specs
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
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

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"

"""
TRAINING PARAMETERS
"""
lr = 0.1
minibatches =10
num_training_epochs = 50
lr_decay = True

new_maxweight_data = False
max_weight_data_type = "rollout" # "rollout" or "enumerate"
train_gnn = True
test_gnn = True
train_mlp = False
test_mlp = False

test_length = 5000
test_rollouts = 3
""" 
ENVIRONMENT PARAMETERS
"""

## Two Set of Two Nodes
# adj = np.array([[0,1,0,0], [1,0,1,0], [0,1,0,1],[0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.3, 0.3, 0.3, 0.3])
# service_dist = "Fixed"
# service_rate = np.array([1, 1, 1, 1])


## Two Nodes
# adj = np.array([[0,1], [1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.4, 0.4])
# service_dist = "Fixed"
# service_rate = np.array([1, 1])

# 4 Node Line Graph
adj = np.array([[0,1,0,0], [1,0,1,0], [0,1,0,1],[0,0,1,0]])
arrival_dist = "Bernoulli"
arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
service_dist = "Fixed"
service_rate = np.array([1, 1, 1, 1])

## 8 Node Line Graph
# adj = np.array([[0,1,0,0,0,0,0,0], [1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0],[0,0,1,0,1,0,0,0],
#                 [0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1], [0,0,0,0,0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4])
# service_dist = "Fixed"
# service_rate = np.array([1, 1, 1, 1, 1, 1, 1, 1])

# ## 8 Node Ring Graph
# adj = np.array([[0,1,0,0,0,0,0,1], [1,0,1,0,0,0,0,0], [0,1,0,1,0,0,0,0],[0,0,1,0,1,0,0,0],
#                 [0,0,0,1,0,1,0,0], [0,0,0,0,1,0,1,0], [0,0,0,0,0,1,0,1], [1,0,0,0,0,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.ones(8) * 0.4
# service_dist = "Fixed"
# service_rate = np.ones(8)

# 9 Node Grid Graph
# adj = np.array([[0,1,0,1,0,0,0,0,0], [1,0,1,0,1,0,0,0,0], [0,1,0,0,0,1,0,0,0],[1,0,0,0,1,0,1,0,0],
#                 [0,1,0,1,0,1,0,1,0], [0,0,1,0,1,0,0,0,1], [0,0,0,1,0,0,0,1,0], [0,0,0,0,1,0,1,0,1], [0,0,0,0,0,1,0,1,0]])
# arrival_dist = "Bernoulli"
# arrival_rate = np.ones(9) * 0.4
# service_dist = "Fixed"
# service_rate = np.ones(9)


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
cfg.training_make_env_kwargs.observation_keys = ["q"]
cfg.training_make_env_kwargs.observation_keys.append("node_priority") # required to differentiate between nodes with the same output embedding

gnn_env_generator = EnvGenerator(input_params=env_params,
                             make_env_keywords = cfg.training_make_env_kwargs.as_dict(),
                             env_generator_seed = 0,
                             cgs = True)


env = gnn_env_generator.sample()

check_env_specs(env)

"""
RUN MAXWEIGHT IF NEW DATA IS NEEDED
"""
maxweight_actor = CGSMaxWeightActor(valid_actions=compute_valid_actions(env))
if new_maxweight_data:
    print("Running MaxWeight Actor")
    if max_weight_data_type == "rollout":
        td = eval_agent(maxweight_actor, gnn_env_generator, max_steps=10_000, rollouts=1)
        #plot maxweight lta
        lta = compute_lta(td["q"].sum(axis=1))
        fig, ax = plt.subplots()
        ax.plot(lta)
        ax.set_xlabel("Time")
        ax.set_ylabel("Queue Length")
        ax.legend()
        ax.set_title("MaxWeight Actor Rollout")
        plt.show()
    else:
        td = create_training_dataset(env, maxweight_actor, q_max = 6)
    pickle.dump(td, open('maxweight_actor_rollout.pkl', 'wb'))

"""
CREATE GNN ACTOR AND CRITIC ARCHITECTURES
"""
node_features = env.observation_spec["observation"].shape[-1]
#policy_module = GCN_Policy_Module(node_features, num_layers = 1)
policy_module = Policy_Module2(node_features, 64, num_layers = 3)


actor = GNN_ActorTensorDictModule(module = policy_module, x_key = "observation", edge_index_key = "adj_sparse", out_keys = ["probs", "logits"])

# do a fake rollout of size 10, setting env.q to be all 1s
q = torch.ones(10, env.num_nodes)
adj_sparse = env.adj_sparse.unsqueeze(0).repeat(10, 1, 1)
node_priority = env.node_priority.repeat(10, 1)
reward = torch.ones(10, 1)

fake_rollout = TensorDict({"q": q, "adj_sparse": adj_sparse, "node_priority": node_priority, "reward": reward})
rollout = env.transform(fake_rollout)
td = actor(rollout)