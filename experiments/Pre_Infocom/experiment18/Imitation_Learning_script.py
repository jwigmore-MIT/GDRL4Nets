import pandas as pd

from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
import os
import torch
from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
from MDP_Solver.SingleHopMDP import SingleHopMDP
import argparse
import os
from analysis_functions import *
from torchrl_development.actors import MDP_module, MDP_actor
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl_development.actors import create_independent_actor_critic
from tensordict import TensorDict
import torch
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage, SamplerWithoutReplacement
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm


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

def plot_state_action_heatmap(df, hold_tuples, ax=None, axis_keys=["Q1", "Q2"], lim=30, action_key = "action", title = ""):
    """
    Want to use plt.imshow to plot the heatmap of the state-action values


    :param df:
    :param hold_tuples:
    :param ax:
    :param axis_keys:
    :param lim:
    :return:
    """
    new_df = df.copy()
    for tup in hold_tuples:
        new_df = new_df[new_df[tup[0]] == tup[1]]
    """
    Want to take the non-hold tuples and plot them as the x and y axis
    What to create a 2D np array of the action values, where dim 1 is the x-axis and dim 2 is the y-axis
    and the value corresponds to the action taken

    """
    x_axis = new_df[axis_keys[0]].unique()
    y_axis = new_df[axis_keys[1]].unique()
    # Limit x and y axis to lim
    x_axis = x_axis[1:lim + 1]
    y_axis = y_axis[1:lim + 1]

    action_values = np.zeros((len(x_axis), len(y_axis)))
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            action_values[i, j] = new_df[(new_df[axis_keys[0]] == x) & (new_df[axis_keys[1]] == y)][action_key]
    ax.imshow(action_values, cmap='plasma')
    ax.set_xticks(range(len(x_axis)))
    ax.set_yticks(range(len(y_axis)))
    ax.set_xticklabels(x_axis)
    ax.set_yticklabels(y_axis)
    ax.set_xlabel(axis_keys[0])
    ax.set_ylabel(axis_keys[1])
    hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
    ax.set_title(hold_names)
    ax.set_xlim(-0.5, len(x_axis) - 0.5)
    ax.set_ylim(-0.5, len(y_axis) - 0.5)
    ax.set_title(title)
    return ax


def plot_state_action_map(df, hold_tuples, ax=None, axis_keys=["Q1", "Q2"], policy_type="MLP", plot_type="Action_Probs",
                          collected_frames=None, lim=30):
    figures = {}
    pt = plot_type
    new_df = df.copy()
    for tup in hold_tuples:
        new_df = new_df[new_df[tup[0]] == tup[1]]
    # fig, ax = plt.subplots(1,1, figsize = (10,10))

    """ Need to convert df to an array where each element is equal to the action taken
    df will have 

    """

    axis_names = axis_keys
    # color should be 1 + the probability the action = 2 -
    # make action a 1-hot vector
    # action_one_hot = pd.get_dummies(new_df["Action"])
    # now make new_df["Action"] the one hot vector by converting action_one_hot to a list of lists
    # new_df["Action"] = action_one_hot.values.tolist()
    if pt == "Action_Probs":
        action_probs = new_df["Action_Probs"].tolist()
        color = [max(x[2] - 1000 * x[0], 0) for x in action_probs]
        # # color = new_df[pt]
        # action_probs = new_df[pt].tolist()
        # Calculate the color as a weighted average of the primary colors based on the action probabilities
        # color = action_probs

    else:
        """
        want the action [0] to be white
                        [1] to be blue
                        [2] to be red
        """
        color = []
        for action in new_df["Action"].tolist():
            if action == 1:
                color.append(0)
            elif action == 2:
                color.append(2)
            else:
                color.append(1)
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c=color, cmap='bwr')
    # sc = ax.imshow
    # fig.colorbar(sc, ax = ax, label = "Action 2 Probability", ticks = [0,1,2])
    if "Y1" in axis_names:
        ax.set_ylim(-0.5, 3)
        ax.set_xlim(-0.5, 3)
    else:
        ax.set_ylim(-0.5, lim)
        ax.set_xlim(-0.5, lim)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
    # combine hold names to create a string
    policy_add = "Deterministic" if pt == "Action" else "Stochastic"
    title = f"{policy_type} {policy_add} Policy for {hold_names}"
    if collected_frames is not None:
        title = title + f" (Collected Frames: {collected_frames})"
    title = hold_names
    ax.set_title(title)


def plot_state_action_map_comparison(df, hold_tuples, ax=None, axis_keys=["Q1", "Q2"], policy_type="MLP",
                                     plot_type="Action_Probs", collected_frames=None, lim=30):
    pass


def get_mdp(param_key, q_max):
    local_path = "Singlehop_Two_Node_Simple_"
    full_path = os.path.join(os.getcwd(), local_path + param_key + ".json")
    base_env_params = parse_env_json(full_path=full_path)
    mdp_name = f"SH_{param_key}"
    env = make_env(base_env_params, terminal_backlog=100)
    mdp = SingleHopMDP(env, name=mdp_name, q_max=q_max, value_iterator='minus')
    # Make Environment
    env = make_env(base_env_params, terminal_backlog=100)

    mdp = SingleHopMDP(env, name=mdp_name, q_max=q_max, value_iterator='minus')

    try:
        mdp.load_tx_matrix(f"tx_matrices/{mdp_name}_qmax{q_max}_discount0.99_computed_tx_matrix.pkl")
    except:
        print("No tx_matrix found!!!")

    try:
        mdp.load_pi_policy(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")
    except:
        print("No PI policy found!!!")

    return mdp


def plot_single_mdp_policy(mdp, lim=20):
    policy_table = mdp.pi_policy.policy_table
    df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])
    df["Action"] = list(policy_table.values())
    df["Action"] = df["Action"].apply(lambda x: np.argmax(x))
    df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plot_state_action_map(df, [("Y1", 1), ("Y2", 1)], ax=ax, axis_keys=["Q1", "Q2"], policy_type="PI",
    #                       plot_type="Action", lim=lim)
    plot_state_action_heatmap(df, [("Y1", 1), ("Y2", 1)], ax=ax, axis_keys=["Q1", "Q2"], lim=lim)
    ax.set_title(f"PI Policy for {mdp.name}")
    plt.show()

def plot_mdp_and_agent_policy(df, suptitle = None):
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]

    plot_state_action_heatmap(df, [("Y1", 1), ("Y2", 1)], ax=ax[0], axis_keys=["Q1", "Q2"], lim=30, action_key = "mdp_action", title = "MDP Policy")
    plot_state_action_heatmap(df, [("Y1", 1), ("Y2", 1)], ax=ax[1], axis_keys=["Q1", "Q2"], lim=30, action_key = "action_prob", title = "Agent Policy")
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.show()

def get_policy_table(mdp, lim = 30):
    policy_table = mdp.pi_policy.policy_table
    df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])


    for i in range(1,mdp.env.N+1):
        df[f"lambda{i}"] = mdp.env.base_env.arrival_rates[i-1]
    df["mask"] = get_mask_from_df(df, mdp)
    # df = df[(df["Q1"] >= 1) & (df["Q2"] >= 1)]
    df["Action"] = list(policy_table.values())
    df["Action"] = df["Action"].apply(lambda x: np.argmax(x))

    # Only include states where Q1 and Q2 are greater are less than lim +1
    df = df[(df["Q1"] <= lim) & (df["Q2"] <= lim)]

    return df

def get_mask_from_df(df, mdp):
    """
    df has keys Q1, Q2, Y1, Y2
    :param df:
    :return:
    """
    masks = []
    for state in df[["Q1", "Q2", "Y1", "Y2"]].values:
        masks.append(mdp.get_actions(state).max(axis =0))
    return masks

def convert_policy_table_to_tensordict(policy_table: pd.DataFrame):
    """
    Convert the policy_table dataframe to a Tensordict with the keys "Q", "Y", "lambda" and values the action
    :param policy_table:
    :return:
    """
    # First convert policy_table["mask"] to a boolean tensor
    mask = np.array([x for x in policy_table["mask"].values], dtype = np.bool_)

    # Create observation tensor which is the Q, Y, and lambda values
    observation = policy_table[["Q1", "Q2", "Y1", "Y2", "lambda1", "lambda2"]].values

    # Convert action from a int in (0,1,2) to a one-hot vector
    one_hot_action = pd.get_dummies(policy_table["Action"])

    td = TensorDict({
        "Q": torch.Tensor(policy_table[["Q1", "Q2"]].values),
        "Y": torch.Tensor(policy_table[["Y1", "Y2"]].values),
        "lambda": torch.Tensor(policy_table[["lambda1", "lambda2"]].values),
        "mask": torch.Tensor(mask).bool(),
        "mdp_action": torch.Tensor(one_hot_action.values),
        "observation": torch.Tensor(observation)
    }, batch_size = len(policy_table))

    return td

def convert_tensordict_to_dataframe(td, temperature = 1.0):
    """
    Convert a tensordict to a dataframe with keys Q1, Q2, Y1, Y2, lambda1, lambda2, mask, mdp_action, action
    :param td:
    :return:
    """
    df = pd.DataFrame(td["observation"].numpy(), columns = ["Q1", "Q2", "Y1", "Y2", "lambda1", "lambda2"])
    # df["mask"] = td["mask"].numpy()]
    # convert action from one-hot array to int

    df["mdp_action"] = td["mdp_action"].argmax(dim =1).numpy()
    df["action"] = td["action"].argmax(dim = 1).numpy()
    df["action_prob"] = logits_to_probs(td["logits"], temperature).detach().numpy()[:,2]
    return df


def logits_to_probs(logits, temperature):
    """
    Convert logits to probabilities
    :param logits:
    :param temperature:
    :return:
    """
    return torch.nn.functional.softmax(logits/temperature, dim = 1)



if __name__ == "__main__":
    mdps = {}
    policy_tables = {}
    policy_tds = {}
    for mdp in ["base", "a", "b", "c", "d"]:
        mdps[mdp] = get_mdp(mdp, 40)
        policy_tables[mdp] = get_policy_table(mdps[mdp], lim  = 20)
        policy_tds[mdp] = convert_policy_table_to_tensordict(policy_tables[mdp])

    # for mdp in mdps.values():
    #     plot_single_mdp_policy(mdp)

    param_key = "d"
    local_path = "Singlehop_Two_Node_Simple_"
    full_path = os.path.join(os.getcwd(), local_path + param_key + ".json")
    base_env_params = parse_env_json(full_path=full_path)
    base_env = make_env(base_env_params, terminal_backlog=100, observe_lambda=True)

    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    action_spec = base_env.action_spec
    N = int(base_env.base_env.N)
    D = int(input_shape[0] / N)

    # Neural Network Parameters
    temperature = 1.0
    actor_depth = 2
    actor_cells = 16
    relu_max = 10


    pmn_agent = create_independent_actor_critic(number_nodes=N,
                                            actor_input_dimension=D,
                                            actor_in_keys=["Q", "Y", "lambda"],
                                            critic_in_keys=["observation"],
                                            action_spec=action_spec,
                                            temperature=temperature,
                                            actor_depth=actor_depth,
                                            actor_cells=actor_cells,
                                            type=1,
                                            network_type="PMN",
                                            relu_max=relu_max, )

    pmn_actor = pmn_agent.get_policy_operator()

    td = pmn_actor(policy_tds[param_key])

    df = convert_tensordict_to_dataframe(td)
    # df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]

    plot_mdp_and_agent_policy(df, suptitle="Before Imitation Learning")

    td["target_action"] = td["mdp_action"]
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
    replay_buffer.extend(td)
    supervised_train(pmn_agent, replay_buffer, num_training_epochs=1000, lr=0.001, loss_fn=nn.CrossEntropyLoss(),
                     weight_decay=0, lr_decay=False, reduce_on_plateau=False,
                     to_plot=["all_losses"], suptitle="Imitation Learning", all_losses=None, all_lrs=None, all_weights=None)

    td = pmn_actor(policy_tds[param_key])
    df = convert_tensordict_to_dataframe(td)
    plot_mdp_and_agent_policy(df, suptitle="After Imitation Learning")




