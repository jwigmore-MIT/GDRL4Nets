import pandas as pd
import numpy as np
import torch
import time
import matplotlib.pyplot as plt


from tensordict import TensorDict
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.envs.transforms import Compose, ActionMask


from modules.torchrl_development.utils.metrics import compute_lta


def evaluate_agent(actor,
                       eval_env_generator,
                       training_envs_ind,
                       pbar,
                       cfg,
                       ):
    log_info = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()
            test_envs_ind = eval_env_generator.context_dicts.keys()
            eval_start = time.time()
            # update pbar to say that we are evaluating
            pbar.set_description("Evaluating")
            # Want to evaluate the policy on all environments from gen_env_generator
            lta_backlogs = {}
            valid_action_fractions = {}
            final_mean_vaf  = {}
            final_mean_lta_backlogs = {}
            normalized_final_mean_lta_backlogs = {}
            num_evals = 0
            eval_tds = []

            num_eval_envs = eval_env_generator.num_envs  # gen_env_generator.num_envs
            for i in eval_env_generator.context_dicts.keys():  # i =1,2 are scaled, 3-6 are general, and 0 is the same as training
                lta_backlogs[i] = []
                valid_action_fractions[i] = []
                eval_env_generator.reseed()
                for n in range(cfg.eval.num_eval_envs):
                    # reset eval_env_generator
                    num_evals += 1
                    # update pbar to say that we are evaluating num_evals/gen_env_generator.num_envs*cfg.eval.num_eval_envs
                    pbar.set_description(
                        f"Evaluating {num_evals}/{eval_env_generator.num_envs * cfg.eval.num_eval_envs} eval environment")
                    eval_env = eval_env_generator.sample(true_ind=i)
                    eval_td = eval_env.rollout(cfg.eval.traj_steps, actor, auto_cast_to_device=True).to('cpu')
                    eval_tds.append(eval_td)
                    eval_backlog = eval_td["next", "backlog"].numpy()
                    eval_lta_backlog = compute_lta(eval_backlog)
                    vaf =  (eval_td["mask"] * eval_td["action"].squeeze()).sum().float() / eval_td["mask"].shape[0]
                    valid_action_fractions[i].append(vaf)
                    lta_backlogs[i].append(eval_lta_backlog)
                final_mean_lta_backlogs[i] = np.mean([t[-1] for t in lta_backlogs[i]])
                # get MaxWeight LTA from gen_env_generator.context_dicts[i]["lta]
                max_weight_lta = eval_env_generator.context_dicts[i]["lta"]
                normalized_final_mean_lta_backlogs[i] = final_mean_lta_backlogs[i] / max_weight_lta
                final_mean_vaf[i] = np.mean(valid_action_fractions[i])
            eval_time = time.time() - eval_start
            log_info.update({f"eval/eval_time": eval_time})
            # add individual final_mean_lta_backlogs to log_info
            for i, lta in final_mean_lta_backlogs.items():
                log_info.update({f"eval/lta_backlog_lambda({i})": lta})
            # add individual normalized_final_mean_lta_backlogs to log_info
            for i, lta in normalized_final_mean_lta_backlogs.items():
                log_info.update({f"eval_normalized/normalized_lta_backlog_lambda({i})": lta})

            # add individual final_mean_vaf to log_info
            for i, vaf in final_mean_vaf.items():
                log_info.update({f"eval/valid_action_fraction_lambda({i})": vaf})


            # log the performanec of the policy on the same environment
            # if all training inds are in test inds then we can do this
            if all([i in test_envs_ind for i in training_envs_ind]):
                training_env_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({f"eval/lta_backlog_training_envs": training_env_lta_backlogs})
                # add the normalized lta backlog for the same environment
                normalized_training_mean_lta_backlogs = np.mean(
                    [normalized_final_mean_lta_backlogs[i] for i in training_envs_ind])
                log_info.update({
                    f"eval_normalized/normalized_lta_backlog_training_envs": normalized_training_mean_lta_backlogs})

            # log the performance of the policy on the non-training environments
            non_training_inds = [i for i in test_envs_ind if i not in training_envs_ind]
            general_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update({"eval/lta_backlog_non_training_envs": general_lta_backlogs})
            # add the normalized lta backlog for the general environments
            normalized_general_lta_backlogs = np.mean(
                [normalized_final_mean_lta_backlogs[i] for i in non_training_inds])
            log_info.update(
                {"eval_normalized/normalized_lta_backlog_non_training_envs": normalized_general_lta_backlogs})

            # log the performance of the policy on all environments
            all_lta_backlogs = np.mean([final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval/lta_backlog_all_envs": all_lta_backlogs})
            # add the normalized lta backlog for all environments
            normalized_all_lta_backlogs = np.mean([normalized_final_mean_lta_backlogs[i] for i in test_envs_ind])
            log_info.update({"eval_normalized/normalized_lta_backlog_all_envs": normalized_all_lta_backlogs})

    return log_info, eval_tds



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
from matplotlib.colors import LinearSegmentedColormap

CDICT = {
          'red':   [(0.0, 1.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)],
        'green': [(0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 1.0, 1.0)]
    }
CUSTOM_CMAP = LinearSegmentedColormap('custom_cmap', CDICT)

def plot_state_action_map(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = None):

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
        color = new_df[pt]
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap = "RdYlBu")
    # fig.colorbar(sc, ax = ax, label = "Action 2 Probability", ticks = [0,1,2])
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
    title = f"{policy_type} {policy_add} Policy for {hold_names}"
    if collected_frames is not None:
        title = title + f" (Collected Frames: {collected_frames})"
    title = hold_names
    ax.set_title(title)
    # fig.tight_layout()
    # figures[f"{hold_names}_{policy_add}"] = fig
    # fig.show()
    return sc, ax

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
def create_action_maps(actor, env, cfg, data_buffer, collected_frames = None, last_train_backlog = None, last_eval_backlog = None):
    from mpl_toolkits.axes_grid1 import ImageGrid

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()

            # Get state_action_map
            state_action_map = create_state_action_map_from_model(actor, env, temp = cfg.agent.temperature, compute_action_prob = True)
            if collected_frames > 0:
                state_freq_map = create_state_freq_from_buffer(data_buffer, env)
            Y_hold_tuples = [[("Y1", 1), ("Y2", 1)],
                                [("Y1", 1), ("Y2", 2)],
                                [("Y1", 2), ("Y2", 1)],
                                [("Y1", 2), ("Y2", 2)]]

            Q_hold_tuples = [[("Q1", 5), ("Q2", 5)],
                                [("Q1", 10), ("Q2", 5)],
                                [("Q1", 5), ("Q2", 10)],
                                [("Q1", 10), ("Q2", 10)]]
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            fig, axes = plt.subplots(2,4, figsize = (20,10))
            for hold_tuples, ax in zip(Y_hold_tuples, axes.flatten()[:4]):
                sc, ax = plot_state_action_map(state_action_map, hold_tuples, ax = ax, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = collected_frames)
            if collected_frames > 0:
                for hold_tuples, ax in zip(Y_hold_tuples, axes.flatten()[4:]):
                    ac, ax = plot_state_freq_map(state_freq_map, hold_tuples, ax = ax, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Frequency", collected_frames = collected_frames)
            cbar_ax = fig.add_axes([0.1, 0.15, 0.03, 0.7])
            fig.colorbar(sc, cax=cbar_ax, label="Action 2 Probability", ticks=[0, 1])
            # if collected_frames > 0:
            #     cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
            #     fig.colorbar(ac, cax=cbar_ax, label="Frequency", ticks=[0, 1])
            # make the title multiple lines starting with Stochastic Policy, Training Steps: {collected_frames}, Last Train Backlog: {last_train_backlog}, Last Eval Backlog: {last_eval_backlog}
            last_train_backlog=np.round(last_train_backlog,decimals=3) if last_train_backlog is not None else last_train_backlog
            last_eval_backlog=np.round(last_eval_backlog,decimals=3) if last_eval_backlog is not None else last_eval_backlog
            title = f"Stochastic Policy @ {collected_frames} time steps\nLast Train Norm Backlog: {last_train_backlog}"
            fig.suptitle(title)

            # Adjust the subplots to make room for the colorbar
            fig.subplots_adjust(right=0.85)

            plt.show()
            return fig

def create_action_maps2(actor, env, cfg, data_buffer, collected_frames = None, last_train_backlog = None, last_eval_backlog = None):
    from mpl_toolkits.axes_grid1 import ImageGrid

    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()

            # Get state_action_map
            state_action_map = create_state_action_map_from_model(actor, env, temp = cfg.agent.temperature, compute_action_prob = True)
            if collected_frames > 0:
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



