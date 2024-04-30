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
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic, create_independent_actor_critic
from torchrl.envs.utils import ExplorationType, set_exploration_type
from importlib import reload


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
def sec_order_train(module, replay_buffer, num_training_epochs=5, lr = 0.01,
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
    weights = []

    def closure(td, loss_fn):
        optimizer.zero_grad()
        td["Q"] = td["Q"].float()
        td["Y"] = td["Y"].float()
        td = module(td)
        loss = loss_fn(td['logits'], td["target_action"])
        loss.backward()
        loss_values.append(loss.item())
        if module.get_policy_operator().module[0].module.__str__() == "MaxWeightNetwork()":
            actor_weights = module.get_policy_operator().module[0].module.get_weights()
            weights.append(actor_weights)
        return loss



    optimizer = optim.LBFGS(module.parameters(), lr=lr)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    for epoch in pbar:
        for mb, td in enumerate(replay_buffer):
            optimizer.step(lambda: closure(td, nn.CrossEntropyLoss()))
            if mb % 10 == 0:
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb, "Loss": loss_values[-1]})
    if plot_losses:
        plots = [(loss_values, "Loss", "Training Loss")]
        plots.append([weights, "Weights", "MaxWeightNetwork Weights"])
        plot_data(plots, suptitle=f"Second Order Training of {module.__class__.__name__}")
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
        def plot_data(plots, suptitle=""):
            num_plots = len(plots)
            fig, axes = plt.subplots(num_plots, 1, sharex=True, figsize=(10, 10))
            if num_plots == 1:
                axes = [axes]


            if all_weights is not None:
                plots.append((all_weights, "Weights", "MaxWeightNetwork Weights"))

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
q_max = 50
num_rollouts = 3
env_generator_seed = 3

# Configure training params
num_training_epochs = 100
lr = 0.01
pickle_string = f"IL_SH1E_nr{num_rollouts}_rl{rollout_length}"

# results storage
results = {}

# MLP Actor Parameters
actor_params = {
  "actor_depth": 2,
  "actor_cells": 64,
}

# Configure Environment Generator
base_env_params = parse_env_json("SH1E.json")

make_env_parameters = {"observe_lambda": False,
                       "device": "cpu",
                       "terminal_backlog": 5000,
                       "inverse_reward": True,
                       "stat_window_size": 100000,
                       "terminate_on_convergence": False,
                       "convergence_threshold": 0.1,
                       "terminate_on_lta_threshold": False}

env_generator = EnvGenerator(base_env_params, make_env_parameters, env_generator_seed)
base_env = env_generator.sample()
input_shape = int(base_env.base_env.N*2)
output_shape = int(base_env.base_env.N+1)



# %% Create MDP Module
mdp = SingleHopMDP(base_env, name = "SH1E", q_max = q_max)
# mdp.load_tx_matrix(f"tx_matrices/SH1Bc_qmax50_discount0.99_computed_tx_matrix.pkl")
mdp.load_VI(f"saved_mdps/SH1Be_qmax60_discount0.99_VI_dict.p")
mdp_actor = MDP_actor(MDP_module(mdp))
# %% Generate Trajectories from mdp_actor
results = pickle.load(open("SH1E_imitation_learning_results.pkl", "rb"))
replay_buffer = pickle.load(open("SH1E_imitation_learning_replay_buffer.pkl", "rb"))
# %% Create the independent Actor
ind_actor = create_independent_actor_critic(
        [input_shape],
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=1,

    )

# %% Train Independent Actor Critic Agent
supervised_train(ind_actor,
                 replay_buffer,
                 num_training_epochs=num_training_epochs,
                 lr=lr,
                 loss_fn=nn.CrossEntropyLoss(),
                 weight_decay=0,
                 to_plot=["all_losses"])
# %% Evaluate the independent actor
results["IND"] = eval_agent(ind_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)



# %% Plot Specific Policies
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    if agent_name not in ["MDP", "MaxWeight", "MLP", "IND", "MWN"]: #["MDP", "MaxWeight", "MLP", "PPO_MLP", "MWN"]
        continue
    linestyle = "-"
    if agent_name == "MDP":
        agent_name = f"VI"
        linestyle = "-"
        color = "tab:blue"
    if agent_name == "MLP":
        agent_name = "Imitation MLP"
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
    if agent_name == "IND":
        agent_name = "Independent Node"
        color = "tab:pink"
        linestyle = "--"
    if agent_name == "MWN_Norm":
        color = "tab:red"
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy ({np.round(policy_results['mean_lta'][-1].item(),2)})", linestyle = linestyle, color = color)
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()





