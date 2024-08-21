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
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic, create_gnn_maxweight_actor_critic
from torchrl.envs.utils import ExplorationType, set_exploration_type


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

rollout_length = 50_000
q_max = 50
num_rollouts = 5
env_generator_seed = 3

# Configure training params
num_training_epochs = 500
lr = 0.0001
#pickle_string = f"IL_SH1E_nr{num_rollouts}_rl{rollout_length}"

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
results["MDP"] = eval_agent(mdp_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)


fig, ax = plt.subplots(1,1)
ax.plot(results["MDP"]["mean_lta"], label = "MDP Policy")
ax.fill_between(range(len(results["MDP"]["mean_lta"])), results["MDP"]["mean_lta"] - results["MDP"]["std_lta"], results["MDP"]["mean_lta"] +results["MDP"]["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

# %% Create ReplayBuffer (Dataset)
td = torch.cat([results["MDP"][i]["td"] for i in range(num_rollouts)])
td["target_action"] = td["action"].int().argmax(dim = 1).long()
replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
replay_buffer.extend(td)

# # %% Create MLP Module
# mlp_agent = create_actor_critic(
#         [input_shape],
#         output_shape,
#         in_keys=["observation"],
#         action_spec=base_env.action_spec,
#         temperature=0.1,
#         actor_depth=actor_params["actor_depth"],
#         actor_cells=actor_params["actor_cells"],
#     )
#
# # %% Create MaxWeightNetwork Agent
# mwn_agent = create_maxweight_actor_critic(input_shape=[input_shape],
#                                                 action_spec=base_env.action_spec, in_keys=["Q", "Y"],
#                                                 temperature=10,
#                                                 init_weights= torch.ones([1,2])
#                                                 )
# # %% Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
results["MaxWeight"] = eval_agent(mw_actor, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# # %% Train MLP Agent
# supervised_train(mlp_agent,
#                  replay_buffer,
#                  num_training_epochs=num_training_epochs,
#                  lr=lr,
#                  loss_fn=nn.CrossEntropyLoss(),
#                  weight_decay=0,
#                  to_plot=["all_losses"])
# results["MLP"] = eval_agent(mlp_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)
# pickle.dump(mlp_agent.state_dict(), open(f"SH1B_trained_agents/imitation_mlp.pkl", "wb"))
#
# # %% For retraining and plotting over all lrs and losses
# mwn_tr_dict = {"all_losses": None, "all_lrs": None, "all_weights": None}
# # %% Train MaxWeightNetwork Agent
# mwn_lr = 0.01
# mwn_tr_dict["all_losses"], mwn_tr_dict["all_lrs"], mwn_tr_dict["all_weights"] = supervised_train(mwn_agent,
#                  replay_buffer,
#                  num_training_epochs=num_training_epochs,
#                  lr=mwn_lr,
#                  loss_fn=nn.CrossEntropyLoss(),
#                  reduce_on_plateau = False,
#                  weight_decay=0,
#                  to_plot = ["all_losses", "all_weights"],
#                  all_losses = mwn_tr_dict["all_losses"],
#                  all_lrs = mwn_tr_dict["all_lrs"],
#                  all_weights = mwn_tr_dict["all_weights"],)
#
# pickle.dump(mwn_agent.state_dict(), open(f"SH1B_trained_agents/imitation_mwn.pkl", "wb"))

#%% Load and test PPO MLP agent
ppo_mlp_agent = create_actor_critic(
        [input_shape],
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        actor_depth=actor_params["actor_depth"],
        actor_cells=actor_params["actor_cells"],
    )
ppo_mlp_agent.load_state_dict(torch.load("SH1E_trained_agents/MLP_E_trained_agent.pt"))
results["PPO_MLP"] = eval_agent(ppo_mlp_agent, env_generator, num_rollouts = num_rollouts, rollout_length = rollout_length)

# %% Plot the results
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy")
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

# %% Plot Specific Policies
fig, ax = plt.subplots(1,1)
for agent_name, policy_results in results.items():
    if agent_name not in ["MDP", "MaxWeight", "MLP", "PPO_MLP"]: #["MDP", "MaxWeight", "MLP", "PPO_MLP", "MWN"]
        continue
    if agent_name == "MDP":
        agent_name = f"VI"
        color = "tab:blue"
    if agent_name == "MLP":
        agent_name = "Imitation MLP"
        color = "tab:green"
    if agent_name == "MaxWeight":
        color = "tab:orange"
    if agent_name == "PPO_MLP":
        agent_name = "PPO MLP"
        color = "tab:olive" # something close to green
    if agent_name == "MWN":
        color = "tab:purple"
    ax.plot(policy_results["mean_lta"], label = f"{agent_name} Policy ({np.round(policy_results['mean_lta'][-1].item(),2)})", color = color)
    ax.fill_between(range(len(policy_results["mean_lta"])), policy_results["mean_lta"] - policy_results["std_lta"], policy_results["mean_lta"] + policy_results["std_lta"], alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

# %% Interpret the results by creating a state-action heatmap for the MLP policy
def create_state_action_map(td, action_key = "action", temp = 0.1):
    state_action_map = {}
    for Q, Y, action, mask in zip(td["Q"], td["Y"], td[action_key], td["mask"]):
        state = tuple(Q.tolist()+ Y.tolist())
        if action_key == "action":
            action = action.int().argmax().item()
        elif action_key == "logits":
            # Make action[i] = -np.inf if mask[i] = False
            action[~mask] = -1e6
            action = torch.softmax(action/temp, dim = 0).numpy().tolist()
        state_action_map[state] = action
    return state_action_map



def create_state_action_map_from_model(model, env, temp = 0.1, compute_action_prob = True):
    """
    Instead of creating the state action map from the td, we can create it from the model by iterating through all states
    and getting the action and logits from the model
    :param model:
    :param env:
    :return:
    """

    N = env.base_env.N
    from tensordict import TensorDict
    Q_ranges = [np.arange(0, 31) for _ in range(env.base_env.N)]
    Y_ranges = [np.arange(0, 3) for _ in range(env.base_env.N)]
    # combine Q_ranges and Y_ranges to create a meshgrid
    ranges = Q_ranges + Y_ranges
    # create a meshgrid of all possible states
    mesh = np.array(np.meshgrid(*ranges)).reshape(2*env.base_env.N, -1).T
    rewards = torch.zeros(mesh.shape[0])
    # create a tensordict from the meshgrid with keys ["Q", "Y"]
    # Create masks, which is a N+1 length tensor, where entries 1:N are 1 if Q[i]*Y[i] > 0, and 0 otherwise, and the first entry is 1 if all others are 0
    mask1 = torch.Tensor(mesh[:,:N] * mesh[:,N:] != 0)  # evaluates to true if the queue is empty or the link is disconnected
    # Mask 2
    mask2 = (mask1 == False).all(dim = 1).unsqueeze(-1) # evaluates to true if all queues are empty or all links are disconnected
    mask = torch.concat([mask2, mask1], dim = 1).bool()
    td = TensorDict({"Q": torch.tensor(mesh[:, :env.base_env.N]), "Y": torch.tensor(mesh[:, env.base_env.N:]), "reward": rewards, "mask": mask}, batch_size=mesh.shape[0])
    td = env.transform[1:](td)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        model.eval()
        td = model(td)
    # create a dataframe from td with columns Q1, ..., QN, Y1, ..., YN, Action, Action_Prob
    df = pd.DataFrame(td["Q"].numpy(), columns = [f"Q{i+1}" for i in range(N)])
    df = pd.concat([df, pd.DataFrame(td["Y"].numpy(), columns = [f"Y{i+1}" for i in range(N)])], axis = 1)
    df["Action"] = td["action"].int().argmax(dim = 1).numpy()
    if compute_action_prob:
        td["logits"][~td["mask"]] = -1e6
        td["Action_Probs"] = torch.softmax(td["logits"]/temp, dim = 1)
        df["Action_Probs"] = td["Action_Probs"].numpy().tolist()
    return df





def compare_state_action_maps(state_action_map1, state_action_map2):
    diffs = {}
    for state, action in state_action_map1.items():
        map2_action = state_action_map2.get(state, "None")
        if map2_action == "None":
            continue
            diffs[state] = (action, "None")
        elif map2_action!= action:
            diffs[state] = (action, map2_action)
    return diffs

def check_if_threshold_policy(state_action_map):
    """
    The state action map has keys (q1, q2, y1, y2) and values (action)
    We want to check if that given any state s_i = (q1, q2, y1, y2), with action a_1, that
    the action for any s_j = (q_1+b, q_2, y_1, y_2) is still a_1, where b is any positive integer
    :param state_action_map:
    :return:
    """
    for state, action in state_action_map.items():
        q1, q2, y1, y2 = state
        if action == 0 or action == 2:
            continue
        for i in range(1, 10):
            new_state = (q1+i, q2, y1, y2)
            if new_state in state_action_map:
                if state_action_map[new_state] != action:
                    return state, new_state, action, state_action_map[new_state]
    # repeat for q2
    for state, action in state_action_map.items():
        q1, q2, y1, y2 = state
        if action == 0 or action == 1:
            continue
        for i in range(1, 10):
            new_state = (q1, q2+i, y1, y2)
            if new_state in state_action_map:
                if state_action_map[new_state] != action:
                    return state, new_state, action, state_action_map[new_state]

    return True

# %% Create state action map for the MLP policy using all the rollouts
for i in range(num_rollouts):
    state_action_map = create_state_action_map(results["PPO_MLP"][i]["td"])
    if i == 0:
        mlp_state_action_map = state_action_map
    else:
        mlp_state_action_map.update(state_action_map)

# %% Create state action map for the MW policy using all the rollouts
for i in range(num_rollouts):
    state_action_map = create_state_action_map(results["MaxWeight"][i]["td"])
    if i == 0:
        mw_state_action_map = state_action_map
    else:
        mw_state_action_map.update(state_action_map)

# %% Compare the state action maps
diffs = compare_state_action_maps(mlp_state_action_map, mw_state_action_map)
# %% Check if the MLP policy is a threshold policy
check_if_threshold_policy(mlp_state_action_map)

# %% Make a copy but remove all states that contains a 0
mlp_state_action_map2 = mlp_state_action_map.copy()
for state, action in mlp_state_action_map.items():
    if 0 in state:
        mlp_state_action_map2.pop(state)


# %% Create a state_logits_map
for i in range(num_rollouts):
    state_logits_map = create_state_action_map(results["PPO_MLP"][i]["td"], "logits")
    if i == 0:
        mlp_state_logits_map = state_logits_map
    else:
        mlp_state_logits_map.update(state_logits_map)

# remove all states containing a 0
mlp_state_logits_map2 = mlp_state_logits_map.copy()
for state, action in mlp_state_logits_map.items():
    if 0 in state:
        mlp_state_logits_map2.pop(state)


# Create a DataFrame for the MLP State Action Map
import pandas as pd
# mlp_state_action_map is a dict with keys as the state tuples, and values as the action
# we want to convert this to a dataframe with columns Q1, Q2, Y1, Y2, Action
mlp_state_action_df = pd.DataFrame(mlp_state_action_map.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
mlp_state_action_df["Action"] = mlp_state_action_map.values()


def plot_state_action_map(df, hold_tuples, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = ["Action"]):
    if isinstance(plot_type, str):
        plot_type = [plot_type]
    for pt in plot_type:
        new_df = df.copy()
        for tup in hold_tuples:
            new_df = new_df[new_df[tup[0]] == tup[1]]
        fig, ax = plt.subplots(1,1, figsize = (5,4))
        axis_names = axis_keys
        # color should be 1 + the probability the action = 2 -
        # make action a 1-hot vector
        # action_one_hot = pd.get_dummies(new_df["Action"])
        # now make new_df["Action"] the one hot vector by converting action_one_hot to a list of lists
        # new_df["Action"] = action_one_hot.values.tolist()
        if pt == "Action_Probs":
            action_probs = new_df["Action_Probs"].tolist()
            color = [max(1 + x[2] - 1000 * x[0], 0) for x in action_probs]
        else:
            color = new_df[pt]
        sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap ="viridis")
        fig.colorbar(sc, ax = ax, label = "Action", ticks = [0,1,2])
        if "Y1" in axis_names:
            ax.set_ylim(-0.5, 3)
            ax.set_xlim(-0.5, 3)
        else:
            ax.set_ylim(-0.5, 25)
            ax.set_xlim(-0.5, 25)
        ax.set_xlabel(axis_names[0])
        ax.set_ylabel(axis_names[1])
        hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
        # combine hold names to create a string
        policy_add = "Deterministic" if pt == "Action" else "Stochastic"
        ax.set_title(f"{policy_type} {policy_add} Policy for {hold_names}")
        fig.tight_layout()
        fig.show()

def plot_state_action_probability_map(df, hold_tuples, policy_type = "MLP"):
    """
    Like plot_state_action_map, but for the state_logits_map where instead of plotting the color corresponding to
    the action, we plot the action probability for action possibilities (0,1,2)
    :param df:
    :param hold_tuples:
    :param policy_type:
    :return:
    """
    new_df = df.copy()
    for tup in hold_tuples:
        new_df = new_df[new_df[tup[0]] == tup[1]]
    fig, ax = plt.subplots(1,1, figsize = (5,4.5))
    axis_names = [x for x in new_df.columns.tolist() if x not in [x[0] for x in hold_tuples] and x != "Action_Probs"]
    # make color that corresponds to the action probabilities
    # color = max(1 + action[2] - 1000*action[0],0)
    action_probs = new_df["Action_Probs"].tolist()
    color = [max(1 + x[2] - 1000*x[0], 0) for x in action_probs]
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap = "viridis")
    if "Y1" in axis_names:
        ax.set_ylim(-0.5, 3)
        ax.set_xlim(-0.5, 3)
    else:
        ax.set_ylim(-0.5, 25)
        ax.set_xlim(-0.5, 25)
    # add legend for the color
    fig.colorbar(sc, ax = ax, label = "Action", ticks = [0,1,2])
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
    # combine hold names to create a string
    ax.set_title(f"{policy_type} Stochastic Policy for {hold_names}")
    fig.tight_layout()
    fig.show()

# %% Plot the state action map for the MLP policy for all possible combinations of Y1, Y2
all_hold_tuples = [(("Y1", 1), ("Y2", 1)),
                   (("Y1", 1), ("Y2", 2)),
                    (("Y1", 2), ("Y2", 1)),
                    (("Y1", 2), ("Y2", 2))]

for hold_tuples in all_hold_tuples:
    plot_state_action_map(mlp_state_action_df, hold_tuples, axis_keys=["Q1", "Q2"],policy_type="MLP")

Q_hold_tuples = [(("Q1", 5), ("Q2", 5)),
                     (("Q1", 5), ("Q2", 10)),
                      (("Q1", 10), ("Q2", 5)),
                      (("Q1", 10), ("Q2", 10))]

for hold_tuples in Q_hold_tuples:
    plot_state_action_map(mlp_state_action_df, hold_tuples, axis_keys=["Y1", "Y2"], policy_type= "MLP")


# Create MDP state_action_map
for i in range(num_rollouts):
    state_action_map = create_state_action_map(results["MDP"][i]["td"])
    if i == 0:
        mdp_state_action_map = state_action_map
    else:
        mdp_state_action_map.update(state_action_map)

mdp_state_action_df = pd.DataFrame(mdp_state_action_map.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
mdp_state_action_df["Action"] = mdp_state_action_map.values()

# %% Plot the state action map for the MDP policy for all possible combinations of Y1, Y2
for hold_tuples in all_hold_tuples:
    plot_state_action_map(mdp_state_action_df, hold_tuples, policy_type= "VI")

for hold_tuples in Q_hold_tuples:
    plot_state_action_map(mdp_state_action_df, hold_tuples, axis_keys = ["Q1", "Q2"], policy_type= "VI")

# Create state action map for the MLP using logits
for i in range(num_rollouts):
    state_logits_map = create_state_action_map(results["PPO_MLP"][i]["td"], "logits")
    if i == 0:
        mlp_state_logits_map = state_logits_map
    else:
        mlp_state_logits_map.update(state_logits_map)

# Create a DataFrame for the MLP State Action Map
mlp_state_action_prob_df = pd.DataFrame(mlp_state_logits_map.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
mlp_state_action_prob_df["Action_Probs"] = [x[1] for x in mlp_state_logits_map.items()]

# %% Plot the state action map for the MLP policy for all possible combinations of Y1, Y2
for hold_tuples in all_hold_tuples:
    plot_state_action_probability_map(mlp_state_action_prob_df, hold_tuples, "MLP")

for hold_tuples in Q_hold_tuples:
    plot_state_action_probability_map(mlp_state_action_prob_df, hold_tuples, "MLP")


# Create a DataFrame for the MaxWeight State Action Map
mw_state_action_df = pd.DataFrame(mw_state_action_map.keys(), columns = ["Q1", "Q2", "Y1", "Y2"])
mw_state_action_df["Action"] = mw_state_action_map.values()

# %% Plot the state action map for the MaxWeight policy for all possible combinations of Y1, Y2
for hold_tuples in all_hold_tuples:
    plot_state_action_map(mw_state_action_df, hold_tuples, axis_keys = ["Q1", "Q2"], policy_type= "MaxWeight", plot_type = ["Action"])

for hold_tuples in Q_hold_tuples:
    plot_state_action_map(mw_state_action_df, hold_tuples, axis_keys = ["Y1", "Y2"], policy_type= "MaxWeight", plot_type = ["Action"])


# # Create state_action_map using ppo_mlp_agent
# env = env_generator.sample()
# ppo_mlp_action_df = create_state_action_map_from_model(ppo_mlp_agent.get_policy_operator(), env)
#
# # %% Plot the state action map for the PPO MLP policy for all possible combinations of Y1, Y2
# for hold_tuples in all_hold_tuples:
#     plot_state_action_map(ppo_mlp_action_df, hold_tuples, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = ["Action", "Action_Probs"])
#
# # create state_action_map using MDP_actor
# mdp_action_df = create_state_action_map_from_model(mdp_actor, env, compute_action_prob = False)

