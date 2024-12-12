

"""
Perform imitation learning on the "GNN" link agent using only a [2,1] Weight matrix --  should converge to [1,-1]
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

from torchrl.envs.utils import check_env_specs

from experiments.MultiClassMultiHopDevelopment.development.backpressure import BackpressureActor
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.objectives.value import GAE


from modules.torchrl_development.utils.metrics import compute_lta
from torchrl.modules import ProbabilisticActor, ActorCriticWrapper
import pickle
from imitation_learning_utils import *




SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
PARAMETERS
"""
gnn_layer = 2
hidden_channels =32

lr = 0.01
minibatches =100
num_training_epochs = 30
lr_decay = True

env_name= "env2"
new_backpressure_data = True
training_data_amount = [10_000, 1]

"""
Get Environment
"""
file_path = f"../envs/{env_name}.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)

# init env
base_env = MultiClassMultiHop(**env_info)
env = TransformedEnv(base_env, MCMHPygLinkGraphTransform(in_keys=["Q"], out_keys=["X"], env=base_env))
env = TransformedEnv(env, SymLogTransform(in_keys=["X"], out_keys=["X"]))

check_env_specs(env)

bp_actor = BackpressureActor(net=env)
if new_backpressure_data:
    print("Running Backpressure Actor")
    training_data = env.rollout(max_steps=training_data_amount[0]*training_data_amount[1], policy=bp_actor)
    lta = compute_lta(-training_data["next","reward"])
    fig, ax = plt.subplots()
    ax.plot(lta)
    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Length")
    ax.legend()
    ax.set_title("BP Actor Rollout")
    plt.show()
else:
    training_data = pickle.load(open(f"{SCRIPT_PATH}/backpressure_data_{env_name}.pkl", "rb"))
pickle.dump(training_data, open(f"{SCRIPT_PATH}/backpressure_data_{env_name}.pkl", "wb"))


### modify td["action"] to add another column which value = 1 if all elements of the row = 0
idle = (training_data["action"] == 0).all(axis=-1).unsqueeze(-1).float()
training_data["action"] = torch.cat([idle, training_data["action"]], dim=-1)
training_data["target_action"]  = training_data["action"].clone().long()

# Create Backpressure GNN Actor
from experiments.MultiClassMultiHopDevelopment.agents.mcmh_link_sage import MCHCGraphSage, GNN_Actor, GNN_Critic

gnn_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=hidden_channels,
                            num_layers=gnn_layer,
                            normalize=False,
                            activate_last_layer=False
                            )

gnn_actor = GNN_Actor(module = gnn_module,
                         feature_key="X", edge_index_key="edge_index",
                                     class_edge_index_key="class_edge_index", out_keys=["logits", "probs"],
                                     init_weight=torch.Tensor([[-2, 4]]))

gnn_critic_module = MCHCGraphSage(in_channels=env.observation_spec["X"].shape[-1],
                            hidden_channels=hidden_channels,
                            num_layers=gnn_layer,
                            normalize=False,
                            activate_last_layer=False
                            )

gnn_critic = GNN_Critic(module = gnn_critic_module,
                            feature_key="X", edge_index_key="edge_index",
                            class_edge_index_key="class_edge_index", out_keys=["state_value"],
                        )


from modules.torchrl_development.agents.utils import  MaskedOneHotCategorical
from torchrl.modules import ActorCriticWrapper
gnn_actor = ProbabilisticActor(gnn_actor,
                           in_keys = ["logits", "mask"],
                           distribution_class= MaskedOneHotCategorical,
                           spec = env.action_spec,
                           default_interaction_type = ExplorationType.RANDOM,
                           return_log_prob=True,
                            )

gnn_agent = ActorCriticWrapper(gnn_actor, gnn_critic)


replay_buffer  = TensorDictReplayBuffer(storage = LazyMemmapStorage(max_size = training_data.shape[0]),
                                            batch_size = training_data.shape[0] // minibatches,
                                            sampler = SamplerWithoutReplacement(shuffle=True))

replay_buffer.extend(training_data)

# all_policy_losses,  all_lrs, all_weights = supervised_train(gnn_agent, replay_buffer,
#                                                         num_training_epochs = num_training_epochs,
#                                                         lr = lr,
#                                                         lr_decay = lr_decay,
#                                                         reduce_on_plateau = False,
#                                                         suptitle = "Imitation Learning with GNN Actor")

all_policy_losses, all_critic_losses,  all_lrs, all_weights = supervised_train_w_critic(gnn_agent, replay_buffer,
                                                        num_training_epochs = num_training_epochs,
                                                        lr = lr,
                                                        lr_decay = lr_decay,
                                                        reduce_on_plateau = False,
                                                        suptitle = "Imitation Learning with GNN Actor")

with torch.no_grad() and set_exploration_type(ExplorationType.MODE):
    td =env.rollout(max_steps=1000, policy=gnn_agent)
bp_actions = torch.zeros_like(td["action"])
for i in range(len(td)):
    bp_action = bp_actor(td[i])["action"]
    # modify bp_action to add another column which value = 1 if all elements of the row = 0
    idle = (bp_action == 0).all(axis=-1).unsqueeze(-1).float()
    bp_action = torch.cat([idle, bp_action], dim=-1)
    bp_actions[i] = bp_action

lta = compute_lta(-td["next","reward"])
fig, ax = plt.subplots()
ax.plot(lta)
ax.set_xlabel("Time")
ax.set_ylabel("Queue Length")
ax.legend()
ax.set_title("Trained GNN Actor Rollout")
fig.show()
