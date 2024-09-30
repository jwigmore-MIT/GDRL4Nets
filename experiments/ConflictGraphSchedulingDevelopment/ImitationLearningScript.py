# import

import os
from torchrl.envs.utils import check_env_specs
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from torchrl.objectives.value import GAE

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

from graph_env_creators import make_line_graph, make_ring_graph, create_grid_graph

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

LOGGING_PATH = os.path.join(os.path.dirname(os.path.dirname(SCRIPT_PATH)), "logs")

device = "cpu"


"""
ENVIRONMENT GENERATING FUNCTIONS
"""


"""
TRAINING PARAMETERS
"""
gnn_layers = 3


lr = 0.01
minibatches =100
num_training_epochs = 30
lr_decay = True

new_maxweight_data = False
training_data_amount = [10_000, 3]
max_weight_data_type = "rollout" # "rollout" or "enumerate"
batch_dataloader = False
train_gnn = True
test_gnn = True
train_mlp = False
test_mlp = False

test_length = 5000
test_rollouts = 3
""" 
ENVIRONMENT PARAMETERS
"""







adj, arrival_dist, arrival_rate, service_dist, service_rate = make_line_graph(4, 0.4, 1)
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
        td = eval_agent(maxweight_actor, gnn_env_generator, max_steps=training_data_amount[0], rollouts=training_data_amount[1], cat = True)
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
policy_module = Policy_Module2(node_features, 32, num_layers = gnn_layers, dropout=0.1)


actor = GNN_ActorTensorDictModule(module = policy_module, x_key = "observation", edge_index_key = "adj_sparse", out_keys = ["probs", "logits"])

value_module = Value_Module(node_features, 32, num_layers = gnn_layers, dropout = 0.1)

critic = GNN_TensorDictModule(module = value_module, x_key="observation", edge_index_key="adj_sparse", out_key="state_value")

actor = ProbabilisticActor(
    actor,
    in_keys=["probs"],
    distribution_class=IndependentBernoulli,
    spec = env.action_spec,
    return_log_prob=True,
    default_interaction_type = ExplorationType.RANDOM
    )

agent = ActorCriticWrapper(actor, critic)

# do a short rollout with the agent
# td = env.rollout(max_steps = 100, policy = agent)

"""
CREATE MLP ACTOR AND CRITIC ARCHITECTURES
"""
mlp_cfg = load_config(os.path.join(SCRIPT_PATH, 'config', 'CGS_MLP_PPO_settings.yaml'))

mlp_cfg.training_make_env_kwargs.observation_keys = ["q", "node_priority"]
mlp_env_generator = EnvGenerator(input_params=env_params,
                                make_env_keywords = mlp_cfg.training_make_env_kwargs.as_dict(),
                                env_generator_seed = 0,
                                cgs = True)

mlp_env = mlp_env_generator.sample()

mlp_agent = create_mlp_actor_critic(
            input_shape = mlp_env.observation_spec["observation"].shape,
            output_shape = mlp_env.action_spec.shape,
            in_keys=["observation"],
            action_spec=env.action_spec,
            actor_depth=3,
            actor_cells=64,
            # dropout = 0.1,
        )

"""
LOAD AND PROCESS TRAINING DATA
"""
training_rollout = pickle.load(open('maxweight_actor_rollout.pkl', 'rb'))
training_rollout["target_action"] = training_rollout["action"].long()



# Apply transformations from the environment to the rollout
training_rollout = env.transform(training_rollout)


# training_rollout2 = maxweight_actor.get_all_mwis_actions(training_rollout)
mw_q_lta = compute_lta(training_rollout["q"].sum(axis=1))
if batch_dataloader:
    # create a list of data from the rollout
    data_list = []
    for i in range(training_rollout["q"].shape[0]):
        data = Data(x=training_rollout["observation"][i],
                    edge_index=training_rollout["adj_sparse"][i],
                    target_action=training_rollout["target_action"][i],
                    )
        data_list.append(data)
    replay_buffer = DataLoader(data_list, batch_size=len(data_list) // minibatches, shuffle=True, drop_last=True)
else:
    replay_buffer  = TensorDictReplayBuffer(storage = LazyMemmapStorage(max_size = training_rollout.shape[0]),
                                            batch_size = training_rollout.shape[0] // minibatches,
                                            sampler = SamplerWithoutReplacement(shuffle=True))

    replay_buffer.extend(training_rollout)

"""
PERFORM IMITATION LEARNING
"""
if train_gnn:
    all_policy_losses, all_critic_losses, all_lrs, all_weights = supervised_train_w_critic(agent, replay_buffer,
                                                        num_training_epochs = num_training_epochs,
                                                        lr = lr,
                                                        lr_decay = lr_decay,
                                                        reduce_on_plateau = False,
                                                        suptitle = "Imitation Learning with GNN Actor")

    # Test GNN agent
    # from modules.torchrl_development.envs.custom_transforms import ObservationNoiseTransform
    # from torchrl.envs.transforms import TransformedEnv
    # env = TransformedEnv(env, ObservationNoiseTransform(noise = 0.0001))
    # evaluate GNN agent
if test_gnn or test_mlp:
    gnn_env_generator.reseed(0)
    mw_tds = eval_agent(maxweight_actor, gnn_env_generator, max_steps=test_length, rollouts = test_rollouts, cat = False)
    mw_q_ltas = torch.stack([compute_lta(td["q"].sum(axis=1)) for td in mw_tds])

if test_gnn:
    gnn_env_generator.reseed(0)
    gnn_tds = eval_agent(agent, gnn_env_generator, max_steps=test_length, rollouts = test_rollouts, cat = False)

    min_len = min([td.shape[0] for td in gnn_tds])
    gnn_q_ltas = torch.stack([compute_lta(td["q"][:min_len,].sum(axis=1)) for td in gnn_tds])

    # plot the results
    fig, ax = plt.subplots()
    ax.plot(gnn_q_ltas.mean(axis = 0),label = "GNN Agent")
    ax.plot(mw_q_ltas.mean(axis=0), linestyle = "--",label = "MaxWeight Agent")
    ax.set_xlabel("Time")
    ax.set_ylabel("LTA Queue Length")
    ax.legend()
    ax.set_title("GNN Agent Rollout")

    plt.show()

    gnn_sa_map = create_action_map([gnn_tds[0]["q"].detach(), gnn_tds[0]["action"].detach(), gnn_tds[0]["logits"].detach()],
                           keys=["q", "a", "l"])


    # import pandas as pd
    # # create a dataframe with td["q"] and td["action"]
    # df = pd.DataFrame(td["q"], columns = [f"q{i}" for i in range(td["q"].shape[1])])
    # df2 = pd.DataFrame(td["action"], columns = [f"a{i}" for i in range(td["action"].shape[1])])
    # df = pd.concat([df, df2], axis = 1)
    #
    # mw_df = pd.DataFrame(training_rollout["q"], columns = [f"q{i}" for i in range(training_rollout["q"].shape[1])])
    # mw_df2 = pd.DataFrame(training_rollout["action"], columns = [f"a{i}" for i in range(training_rollout["action"].shape[1])])
    # mw_df = pd.concat([mw_df, mw_df2], axis = 1)

"""
REPEAT FOR MLP AGENT
"""
if train_mlp:
    training_rollout = pickle.load(open('maxweight_actor_rollout.pkl', 'rb'))
    training_rollout["target_action"] = training_rollout["action"].long()
    # Apply transformations from the environment to the rollout
    training_rollout = mlp_env.transform(training_rollout)


    replay_buffer  = TensorDictReplayBuffer(storage = LazyMemmapStorage(max_size = training_rollout.shape[0]),
                                            batch_size = training_rollout.shape[0]//minibatches,
                                            sampler = SamplerWithoutReplacement(shuffle=True, drop_last=True))

    replay_buffer.extend(training_rollout)

    all_losses, all_lrs, _ = supervised_train(mlp_agent, replay_buffer,
                                                        num_training_epochs = num_training_epochs,
                                                        lr = lr,
                                                        lr_decay = True,
                                                        reduce_on_plateau = False,
                                                        to_plot = ["all_losses", "all_lrs"],
                                                        suptitle = "Imitation Learning with MLP Actor")

    # test MLP agent
if test_mlp:
    mlp_env_generator.reseed(0)
    mlp_tds = eval_agent(mlp_agent, mlp_env_generator, max_steps=test_length, rollouts = test_rollouts, cat = False)
    mlp_env_generator.reseed(0)
    mw_tds2 = eval_agent(maxweight_actor, mlp_env_generator, max_steps=test_length, rollouts=test_rollouts, cat=False)
    mw_q_ltas2 = torch.stack([compute_lta(td["q"].sum(axis=1)) for td in mw_tds2])
    min_len = min([td.shape[0] for td in mlp_tds])
    mlp_q_ltas = torch.stack([compute_lta(td["q"][:min_len, ].sum(axis=1)) for td in mlp_tds])

    # plot the results
    """
    PLOT THE RESULTING QUEUE LENGTHS AS A FUNCTION OF TIME
    """
    fig, ax = plt.subplots()
    ax.plot(mlp_q_ltas.mean(axis = 0), label = "MLP Agent")
    ax.plot(mw_q_ltas2.mean(axis =0), linestyle= "--", label = "MaxWeight Agent")
    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Length")
    ax.legend()
    ax.set_title("MLP Agent Rollout")
    plt.show()
if test_mlp and test_gnn:
    # Plot both MaxWeight ltas
    fig, ax = plt.subplots()
    ax.plot(mw_q_ltas.mean(axis = 0), label = "MaxWeight GNN Environment")
    ax.plot(mw_q_ltas2.mean(axis = 0), linestyle = '--', label = "MaxWeight MLP Environment")
    ax.set_xlabel("Time")
    ax.set_ylabel("Queue Length")
    ax.legend()
    ax.set_title("MaxWeight Rollouts")
    plt.show()

