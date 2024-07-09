import numpy as np
import pandas as pd
import torch
from tensordict import TensorDict
from torchrl.envs import Compose, ActionMask
from torchrl.envs import ExplorationType, set_exploration_type
import matplotlib.pyplot as plt


def create_state_action_map_from_model(model, env, temp = 0.1, compute_action_prob = True, Q_max = 30, Y_max = 2):
    """
    Instead of creating the state action map from the td, we can create it from the model by iterating through all states
    and getting the action and logits from the model
    :param model:
    :param env:
    :return:
    """

    N = env.base_env.N
    Q_ranges = [np.arange(0, Q_max+1) for _ in range(env.base_env.N)]
    Y_ranges = [np.arange(0, Y_max+1) for _ in range(env.base_env.N)]
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
    # Get Lambda
    lambdas = torch.ones([mesh.shape[0], N])*torch.Tensor(env.base_env.arrival_rates)
    td = TensorDict({"Q": torch.tensor(mesh[:, :env.base_env.N]), "Y": torch.tensor(mesh[:, env.base_env.N:]), "lambda": lambdas, "reward": rewards, "mask": mask}, batch_size=mesh.shape[0])
    transforms = Compose(*[t.clone() for t in env.transform if not isinstance(t, ActionMask)])
    td = transforms(td)
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
        df["logits"] = td["logits"].numpy().tolist()
    if "ReverseSignTransform(keys=['Y'])" in [str(x) for x in env.transform]:
        for column in df.columns:
            if "Y" in column:
                df[column] = -df[column]
    return df


def plot_state_action_map(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = None, lim = 30):

    figures = {}
    pt = plot_type
    new_df = df.copy()
    for tup in hold_tuples:
        new_df = new_df[new_df[tup[0]] == tup[1]]
    # fig, ax = plt.subplots(1,1, figsize = (10,10))
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
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap = 'bwr')
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
    # fig.tight_layout()
    # figures[f"{hold_names}_{policy_add}"] = fig
    # fig.show()
    # return sc, ax


def create_action_maps2(actor, env, cfg, data_buffer = None, collected_frames = None, last_train_backlog = None, last_eval_backlog = None):
    from mpl_toolkits.axes_grid1 import ImageGrid

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()

            # Get state_action_map
            state_action_map = create_state_action_map_from_model(actor, env, temp = cfg.agent.temperature, compute_action_prob = True)
            if collected_frames > 0 and data_buffer is not None:
                state_freq_map = create_state_freq_from_buffer(data_buffer, env)
            Y_hold_tuples = [[("Y1", 1), ("Y2", 1)],
                                [("Y1", 1), ("Y2", 2)],
                                [("Y1", 2), ("Y2", 1)],
                                [("Y1", 2), ("Y2", 2)]]


            fig = plt.figure(figsize=(20,10))
            grid_top = ImageGrid(fig, 211, nrows_ncols=(1, 4), axes_pad=0.1, cbar_location="right", cbar_mode="single", cbar_pad=0.1)
            grid_bottom = ImageGrid(fig, 212, nrows_ncols=(1, 4), axes_pad=0.1, cbar_location="right", cbar_mode="single", cbar_pad=0.1)
            for hold_tuples, ax in zip(Y_hold_tuples, grid_top):
                sc, ax = plot_state_action_map(state_action_map, hold_tuples, ax = ax, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = collected_frames)
            if collected_frames > 0:
                for hold_tuples, ax in zip(Y_hold_tuples, grid_bottom):
                    ac, ax = plot_state_freq_map(state_freq_map, hold_tuples, ax = ax, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Frequency", collected_frames = collected_frames)
            grid_top.cbar_axes[0].colorbar(sc, label="Action 2 Probability", ticks=[0, 1])
            if collected_frames > 0:
                grid_bottom.cbar_axes[0].colorbar(ac, label="Frequency", ticks=[0, 1])
            # make the title multiple lines starting with Stochastic Policy, Training Steps: {collected_frames}, Last Train Backlog: {last_train_backlog}, Last Eval Backlog: {last_eval_backlog}
            last_train_backlog=np.round(last_train_backlog,decimals=3) if last_train_backlog is not None else last_train_backlog
            title = f"Stochastic Policy @ {collected_frames} time steps\nLast Train Norm Backlog: {last_train_backlog}"
            fig.suptitle(title)

            # Adjust the subplots to make room for the colorbar
            fig.subplots_adjust(right=0.85)

            plt.show()
            return fig


def create_state_freq_from_buffer(data_buffer, env, Q_max = 30, Y_max =2):
    """
    Creates a dataframe of all possible states, and measures the state frequency using all data in the data buffer
    :param data_buffer:
    :param Q_max:
    :param Y_max:
    :return:
    """
    N = env.base_env.N
    # Q_ranges = [np.arange(0, Q_max+1) for _ in range(env.base_env.N)]
    # Y_ranges = [np.arange(0, Y_max+1) for _ in range(env.base_env.N)]
    # # get lambda ranges from the data buffer,
    # lambda_ranges = [data_buffer["lambda"][i,:].unique(dim =0).numpy() for i in range(N)]
    # ranges = Q_ranges + Y_ranges
    # # create a meshgrid of all possible states
    # mesh = np.array(np.meshgrid(*ranges)).reshape(2 * env.base_env.N, -1).T

    sample = data_buffer.sample()
    # create a dataframe from sample["Q"], sample["Y"], and sample["lambda"] with columns Q1, ..., QN, Y1, ..., YN, Lambda1, ..., LambdaN
    df = pd.DataFrame(sample["Q"].numpy(), columns = [f"Q{i+1}" for i in range(N)])
    df = pd.concat([df, pd.DataFrame(sample["Y"].numpy(), columns = [f"Y{i+1}" for i in range(N)])], axis = 1)
    df = pd.concat([df, pd.DataFrame(sample["lambda"].numpy(), columns = [f"Lambda{i+1}" for i in range(N)])], axis = 1)
    if "ReverseSignTransform(keys=['Y'])" in [str(x) for x in env.transform]:
        for column in df.columns:
            if "Y" in column:
                df[column] = -df[column]
    # Get state visitation frequency by counting the number of times each state appears in the sample
    state_frequency = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'Frequency'})
    return state_frequency

def plot_state_freq_map(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Frequency", collected_frames = None):
    new_df = df.copy()
    for tup in hold_tuples:
        new_df = new_df[new_df[tup[0]] == tup[1]]
    # fig, ax = plt.subplots(1,1, figsize = (10,10))
    axis_names = axis_keys
    color = new_df["Frequency"]/df.__len__()*100
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap = "Greens")
    if "Y1" in axis_names:
        ax.set_ylim(-0.5, 3)
        ax.set_xlim(-0.5, 3)
    else:
        ax.set_ylim(-0.5, 25)
        ax.set_xlim(-0.5, 25)
    ax.set_xlabel(axis_names[0])
    ax.set_ylabel(axis_names[1])
    hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
    title = hold_names
    ax.set_title(title)
    return sc, ax
