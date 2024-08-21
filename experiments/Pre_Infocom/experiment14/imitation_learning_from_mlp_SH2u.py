# %%
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from torchrl_development.actors import MDP_module, MDP_actor
import torch
import torch.nn as nn
from torch.optim import Adam, RMSprop
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import pickle
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic, create_independent_actor_critic, create_weight_function_network
from torchrl.envs.utils import ExplorationType, set_exploration_type
import json
import os


if __file__ is not None:
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
else:
    SCRIPT_PATH = os.path.join(os.getcwd(), "experiments/experiment14")


# %%
def plot_data(plots, suptitle=""):
    num_plots = len(plots)
    fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
    if num_plots == 1:
        axes = [axes]


    for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        if title == "MaxWeightNetwork Weights":
            data = torch.stack(data).squeeze().detach().numpy()
            print("Weights shape: ", data.shape)
            for j in range(data.shape[1]):
                if j == 0:
                    continue
                ax.plot(data[:, j], label=f"W{j}")
            ax.legend()
        else:
            ax.plot(data)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    ax.set_xlabel("Minibatch")

    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.show()
def sec_order_train(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr = 1,
                    plot_losses = True):
    """
    Second order optimization for training a module with replay buffer
    :param module:
    :param replay_buffer:
    :param in_keys:
    :param num_training_epochs:
    :param lr:
    :return:
    """

    loss_values = []

    def closure(td, loss_fn):
        optimizer.zero_grad()
        td["Q"] = td["Q"].float()
        td["Y"] = td["Y"].float()
        td = module(td)
        loss = loss_fn(td['logits'], td["target_action"])
        loss.backward()
        loss_values.append(loss.item())
        return loss



    optimizer = optim.LBFGS(module.parameters(), lr=0.01)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    for epoch in pbar:
        for mb, td in enumerate(replay_buffer):
            optimizer.step(lambda: closure(td, nn.CrossEntropyLoss()))
            if mb % 10 == 0:
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb, "Loss": loss_values[-1]})
    if plot_losses:
        fig, ax = plt.subplots(1, 1)
        ax.plot(loss_values)
        ax.set_xlabel("Minibatch")
        ax.set_ylabel("Loss")
        fig.show()
    return loss_values



def supervised_train(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.CrossEntropyLoss(), weight_decay = 1e-5, lr_decay = False, reduce_on_plateau = False,
                to_plot = ["all_losses"], suptitle = "",all_losses = None, all_lrs = None, all_weights = None):
    loss_fn = loss_fn
    # optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    if reduce_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4, verbose=True)

    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    if all_losses is None:
        all_losses = []
    if all_lrs is None:
        all_lrs = []
    if all_weights is None:
        all_weights = []
    last_n_losses = []
    for epoch in pbar:
        # add learning rate decay
        if not reduce_on_plateau and lr_decay:
            alpha = 1 - (epoch / num_training_epochs)
            for group in optimizer.param_groups:
                group["lr"] = lr * alpha

        for mb, td in enumerate(replay_buffer):
            optimizer.zero_grad()
            td["Q"] = td["Q"].float()
            td["Y"] = td["Y"].float()
            td = module(td)
            loss = loss_fn(td['logits'], td["target_action"])
            loss.backward(retain_graph = False)
            optimizer.step()
            all_losses.append(loss.detach().item())
            all_lrs.append(optimizer.param_groups[0]["lr"])
            if module.get_policy_operator().module[0].module.__str__() == "MaxWeightNetwork()":
                actor_weights = module.get_policy_operator().module[0].module.get_weights()
                all_weights.append(actor_weights)
            if reduce_on_plateau:
                scheduler.step(loss)
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.detach().item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
    if len(to_plot) > 0:
        # check if all_weights is empty
        # def plot_data(plots, suptitle=""):
        #     num_plots = len(plots)
        #     fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
        #     if num_plots == 1:
        #         axes = [axes]
        #
        #
        #     if all_weights is not None:
        #         plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        #
        #     for i, ((data, ylabel, title), ax) in enumerate(zip(plots, axes)):
        #         if title == "MaxWeightNetwork Weights":
        #             data = torch.stack(data).squeeze().detach().numpy()
        #             print("Weights shape: ", data.shape)
        #             for j in range(data.shape[1]):
        #                 if j == 0:
        #                     continue
        #                 ax.plot(data[:, j], label=f"W{j}")
        #             ax.legend()
        #         else:
        #             ax.plot(data)
        #         ax.set_ylabel(ylabel)
        #         ax.set_title(title)
        #
        #     if num_plots == 2:
        #         ax[1].set_xlabel("Minibatch")
        #
        #     fig.suptitle(suptitle)
        #     fig.tight_layout()
        #     fig.show()

        plots = []
        if "all_losses" in to_plot:
            plots.append((all_losses, "Loss", "Training Loss"))
        if "all_lrs" in to_plot:
            plots.append((all_lrs, "Learning Rate", "Learning Rate Schedule"))
        if "all_weights" in to_plot:
            plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))
        plot_data(plots, suptitle=suptitle)
    return all_losses, all_lrs, all_weights
        # stop training if loss converges

def eval_agent(agent, env_generator, num_rollouts = 3, rollout_length = 10000):
    results = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        for i in range(num_rollouts):
            env = env_generator.sample()
            td = env.rollout(policy=agent, max_steps = rollout_length)
            results[i] = {"td": td, "lta": compute_lta(td["backlog"])}
        results["mean_lta"] = torch.stack([torch.tensor(results[i]["lta"]) for i in range(num_rollouts)]).mean(dim = 0)
        results["std_lta"] = torch.stack([torch.tensor(results[i]["lta"]) for i in range(num_rollouts)]).std(dim = 0)
        env_generator.reseed()
    return results

# %%



rollout_length = 30000
num_rollouts = 3
env_generator_seed = 5031997

# Configure training params
num_training_epochs = 100
lr = 0.0001
pickle_string = f"MLP_IL_SH2u_nr{num_rollouts}_rl{rollout_length}"

# results storage
results = {}

# MLP Actor Parameters
actor_params = {
  "actor_depth": 2,
  "actor_cells": 64,
}

# Configure Environment Generator
# base_env_params = parse_env_json("SH1E.json")
context_set = json.load(open(os.path.join(SCRIPT_PATH, "SH2u_context_set_10_03211514.json"), 'rb'))['context_dicts']
env_params = context_set['0']["env_params"]
env_params["lta"] = context_set['0']["lta"]

make_env_parameters = {"observe_lambda": False,
                       "device": "cpu",
                       "terminal_backlog": 5000,
                       "inverse_reward": True,
                       "stat_window_size": 100000,
                       "terminate_on_convergence": False,
                       "convergence_threshold": 0.1,
                       "terminate_on_lta_threshold": False}

env_generator = EnvGenerator(env_params, make_env_parameters, env_generator_seed)
base_env = env_generator.sample()
input_shape = int(base_env.base_env.N*2)
output_shape = int(base_env.base_env.N+1)



# %% Create MLP Module
mlp_agent = create_actor_critic(
        [input_shape],
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        actor_depth=actor_params["actor_depth"],
        actor_cells=actor_params["actor_cells"],
)

# Load MLP agent
mlp_agent.load_state_dict(torch.load(
    os.path.join(SCRIPT_PATH,"SH2u_trained_agents/MLP_SH2u_trained_agent.pt")))

# %% Collect Trajectories under MLP Policy
try:
    results = pickle.load(open(f"SH2u_agent_results.pkl", "rb"))
except:
    results = {}
    results["MLP"] = eval_agent(mlp_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# %%
# %% Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
# %%
if "MaxWeight" not in results.keys():
    results["MaxWeight"] = eval_agent(mw_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
#%% Plot results thus far
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy")
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()


# %% Create ReplayBuffer (Dataset)
td = torch.cat([results["MLP"][i]["td"] for i in range(num_rollouts)])
td["target_action"] = td["action"].int().argmax(dim = 1).long()
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
replay_buffer.extend(td)

# %% Create MaxWeightNetwork Agent
mwn_agent = create_maxweight_actor_critic(input_shape=[input_shape],
                                                action_spec=base_env.action_spec, in_keys=["Q", "Y"],
                                                temperature=10,
                                                init_weights= torch.ones([1,int(input_shape/2)])
                                                )


# %% For retraining and plotting over all lrs and losses
mwn_tr_dict = {"all_losses": None, "all_lrs": None, "all_weights": None}
# %% Train MaxWeightNetwork Agent
try:
    state_dict = pickle.load(open(f"SH2u_trained_agents/imitation_mwn.pkl", "rb"))
    mwn_agent.load_state_dict(state_dict)
except:
    mwn_lr = 0.01
    mwn_tr_dict["all_losses"], mwn_tr_dict["all_lrs"], mwn_tr_dict["all_weights"] = supervised_train(mwn_agent,
                     replay_buffer,
                     num_training_epochs=num_training_epochs,
                     lr=mwn_lr,
                     loss_fn=nn.CrossEntropyLoss(),
                     reduce_on_plateau = False,
                     weight_decay=0,
                     to_plot=["all_losses", "all_weights"],
                     all_losses = mwn_tr_dict["all_losses"],
                     all_lrs = mwn_tr_dict["all_lrs"],
                     all_weights = mwn_tr_dict["all_weights"],)

    pickle.dump(mwn_agent.state_dict(), open(f"SH2u_trained_agents/imitation_mwn.pkl", "wb"))
# %% Evaluate MWN Agent
if "MWN" not in results.keys():
    results["MWN"] = eval_agent(mwn_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)


#%% Second Order Training of MaxWeightNetwork Agent
# mwn_agent2 = create_maxweight_actor_critic(input_shape=[input_shape], output_shape=output_shape,
#                                                 action_spec=base_env.action_spec, in_keys=["Q", "Y"],
#                                                 temperature=10
#                                                 )
# losses = sec_order_train(mwn_agent2, replay_buffer, num_training_epochs=num_training_epochs)
# pickle.dump(mwn_agent2.state_dict(), open(f"SH1B_trained_agents/imitation_2ndOrder_mwn.pkl", "wb"))
# #%% Evaluate MWN Agent2
# results["MWN2"] = eval_agent(mwn_agent2, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)

# #%% Load and test PPO MLP agent
# ppo_mlp_agent = create_actor_critic(
#         [input_shape],
#         output_shape,
#         in_keys=["observation"],
#         action_spec=base_env.action_spec,
#         temperature=0.1,
#         actor_depth=actor_params["actor_depth"],
#         actor_cells=actor_params["actor_cells"],
#     )
# ppo_mlp_agent.load_state_dict(torch.load("SH1B_trained_agents/ppo_mlp.pt"))
# results["PPO_MLP"] = eval_agent(ppo_mlp_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# %% Create Independent Actor Critic
# ind_actor_depth = 2
# ind_actor_width = 8
#
# ind_actor = create_independent_actor_critic(
#         [input_shape],
#         in_keys=["observation"],
#         action_spec=base_env.action_spec,
#         temperature=1,
#         actor_depth=ind_actor_depth,
#         actor_cells=ind_actor_width,
#         type = 1,
#     )
# ind_tr_dict = {"all_losses": None, "all_lrs": None}
#
# # %% Train Independent Actor Critic Agent
# ind_lr = 0.01
# ind_tr_dict["all_losses"], ind_tr_dict["all_lrs"], _ = supervised_train(ind_actor,
#                  replay_buffer,
#                  num_training_epochs=num_training_epochs,
#                  lr=ind_lr,
#                  loss_fn=nn.CrossEntropyLoss(),
#                  weight_decay=0,
#                  to_plot=["all_losses", "all_lrs"],
#                  all_losses = ind_tr_dict["all_losses"],
#                  all_lrs = ind_tr_dict["all_lrs"],
#                  )
# # %% Evaluate the independent actor
# results["IND"] = eval_agent(ind_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)

# %% Get all arrival and service rates
arrival_rates = torch.Tensor([env_params["X_params"][i]['arrival_rate'] for i in env_params["X_params"].keys()]).unsqueeze(-1)
service_rates = torch.Tensor([env_params["Y_params"][i]['service_rate'] for i in env_params["Y_params"].keys()]).unsqueeze(-1)

# %% Create Weight Function Network
wfn = create_weight_function_network(input_shape=[input_shape],
                                     in_keys=["Q", "Y"],
                                     action_spec = base_env.action_spec, arrival_rates=arrival_rates, service_rates=service_rates)

# %% Train Weight Function Network
wfn_lr = 0.01
wfn_tr_dict = {"all_losses": None, "all_lrs": None}
wfn_tr_dict["all_losses"], wfn_tr_dict["all_lrs"], _ = supervised_train(wfn,
                replay_buffer,
                 num_training_epochs=num_training_epochs,
                 lr=wfn_lr,
                 loss_fn=nn.CrossEntropyLoss(),
                 weight_decay=0,
                 to_plot=["all_losses", "all_lrs"],
                 all_losses = wfn_tr_dict["all_losses"],
                 all_lrs = wfn_tr_dict["all_lrs"],
                 )
# %%
results["WFN"] = eval_agent(wfn, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# %% Create Independent Actor Critic
# ind_actor_depth = 2
# ind_actor_width = 16
#
# ind_actor2 = create_independent_actor_critic(
#         [input_shape],
#         in_keys=["observation"],
#         action_spec=base_env.action_spec,
#         temperature=1,
#         actor_depth=ind_actor_depth,
#         actor_cells=ind_actor_width,
#         type = 2,
#
#     )
# ind_tr_dict = {"all_losses": None, "all_lrs": None}
#
# # %% Train Independent Actor Critic Agent
# ind_lr = 0.1
# ind_tr_dict["all_losses"], ind_tr_dict["all_lrs"], _ = supervised_train(ind_actor2,
#                     replay_buffer,
#                     num_training_epochs=num_training_epochs,
#                     lr=ind_lr,
#                     loss_fn=nn.CrossEntropyLoss(),
#                     weight_decay=0,
#                     to_plot=["all_losses", "all_lrs"],
#                     all_losses = ind_tr_dict["all_losses"],
#                     all_lrs = ind_tr_dict["all_lrs"],
#                     )
# # %%
# results["IND2"] = eval_agent(ind_actor2, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# %% Plot the results
# fig, ax = plt.subplots(1,1)
# for agent_name, policy_results in results.items():
#     ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy")
#     ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
# ax.set_xlabel("Time")
# ax.set_ylabel("Backlog")
# ax.legend()
# fig.show()

# %% Save Results
pickle.dump(results, open(f"SH2u_agent_results.pkl", "wb"))

# %% Plot Specific Policies
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    if agent_name not in ["MDP", "MaxWeight", "MLP", "IND", "WFN", "MWN", "MWN2"]: #["MDP", "MaxWeight", "MLP", "PPO_MLP", "MWN"]
        continue
    linestyle = "-"
    if agent_name == "MDP":
        agent_name = f"VI"
        linestyle = "-"
        color = "tab:blue"
    if agent_name == "MLP":
        agent_name = "PPO MLP"
        linestyle = "-"
        color = "tab:green"
    if agent_name == "MaxWeight":
        color = "tab:orange"
        linestyle = "-"
    if agent_name == "PPO_MLP":
        agent_name = "PPO MLP"
        color = "tab:olive" # something close to green
        linestyle = "-"
    if agent_name == "MWN":
        color = "tab:purple"
        agent_name = "IL MWN"
    if agent_name == "MWN2":
        color = "tab:purple"
        linestyle = "*"
        agent_name = "2O-IL MWN"
    if agent_name == "IND":
        agent_name = "Independent Node"
        color = "tab:pink"
        linestyle = "--"
    if agent_name == "IND2":
        agent_name = "Product Independent Node"
        color = "tab:pink"
        linestyle = "-."
    if agent_name == "MWN_Norm":
        color = "tab:red"
    if agent_name == "WFN":
        agent_name = "Weight Function Network"
        color = "tab:brown"

    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy ({np.round(policy_results['mean_lta'][-1].item(),2)})", linestyle = linestyle, color = color)
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()


