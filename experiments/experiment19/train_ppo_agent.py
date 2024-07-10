# %%
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.utils.configuration import load_config
from torchrl_development.actors import create_maxweight_actor_critic, create_actor_critic
from torchrl_development.utils.configuration import smart_type
import argparse
import os
from datetime import datetime
from torchrl_development.envs.env_generators import parse_env_json
import torch
import json
from torchrl_development.envs.env_generators import make_env
from torchrl_development.actors import MaxWeightActor
from torchrl_development.utils.metrics import compute_lta
from torchrl.modules import MLP,QValueActor
from torchrl_development.actors import MinQValueActor
from torchrl.modules import EGreedyModule
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate
from torchrl.collectors import MultiaSyncDataCollector, MultiSyncDataCollector, SyncDataCollector
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.data import TensorDictReplayBuffer, SamplerWithoutReplacement, LazyMemmapStorage
from tensordict import TensorDict
from torchrl.data import CompositeSpec
from tensordict.nn import TensorDictSequential
import tempfile
import wandb
from torchrl.record.loggers import get_logger, generate_exp_name
import time
import tqdm
import numpy as np
from torchrl.objectives.value import GAE
from torchrl.objectives import ClipPPOLoss
from torchrl_development.MultiEnvSyncDataCollector import MultiEnvSyncDataCollector
import matplotlib.pyplot as plt
from torchrl_development.utils.configuration import make_serializable
import monotonicnetworks as lmn
from torchrl_development.SMNN import PureMonotonicNeuralNetwork as PMN, MultiLayerPerceptron as MLP, ReLUUnit, ExpUnit, FCLayer_notexp,ReLUnUnit
from torchrl.envs import ParallelEnv
from torchrl.trainers.helpers import make_collector_onpolicy
from torchrl.collectors.utils import  split_trajectories
import pandas as pd
from copy import deepcopy
from tensordict import TensorDict
from torchrl.envs import Compose, ActionMask
from torchrl_development.actors import create_independent_actor_critic
from torchrl_development.SMNN import DeepSetScalableMonotonicNeuralNetwork as DSMNN
""" Script Description
This script is used to experiment with using DQN to solve for the optimal policy of SingleHop environments.
Its mostly based off the dqn_atary.py script from torchrl library


"""
# %%

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

def evaluate_dqn_agent(actor,
                       eval_env_generator,
                       training_envs_ind,
                       pbar,
                       cfg,
                       device):
    log_info = {}
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):

            actor.eval()
            test_envs_ind = eval_env_generator.context_dicts.keys()
            # # Get state_action_map
            # state_action_map = create_state_action_map_from_model(actor, eval_env_generator.sample(), temp = cfg.agent.temperature, compute_action_prob = True)
            # Y_hold_tuples = [[("Y1", 1), ("Y2", 1)],
            #                     [("Y1", 1), ("Y2", 2)],
            #                     [("Y1", 2), ("Y2", 1)],
            #                     [("Y1", 2), ("Y2", 2)]]
            #
            # Q_hold_tuples = [[("Q1", 5), ("Q2", 5)],
            #                     [("Q1", 10), ("Q2", 5)],
            #                     [("Q1", 5), ("Q2", 10)],
            #                     [("Q1", 10), ("Q2", 10)]]
            # state_action_figures = {}
            # for hold_tuples in Y_hold_tuples:
            #     figures = plot_state_action_map(state_action_map, hold_tuples, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = ["Action_Probs"])
            #     state_action_figures.update(figures)
            # # for hold_tuples in Q_hold_tuples:
            # #     figures = plot_state_action_map(state_action_map, hold_tuples, axis_keys = ["Y1", "Y2"], policy_type = "MLP", plot_type = ["Action_Probs"])
            # #     state_action_figures.update(figures)

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


# %%

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def train_ppo_agent(cfg, training_env_generator, eval_env_generator, device, logger = None, disable_pbar = False):


    base_env = training_env_generator.sample()
    training_env_generator.clear_history()
    # Create DQN Agent
    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n
    action_spec = base_env.action_spec
    N = int(base_env.base_env.N)
    D = int(input_shape[0]/N)
   # create actor and critic
    if getattr(cfg.agent, "actor_type", "MLP") == "MLP":
        agent = create_actor_critic(
            input_shape,
            output_shape,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=cfg.agent.temperature,
            actor_depth=cfg.agent.hidden_sizes.__len__(),
            actor_cells=cfg.agent.hidden_sizes[-1],
        )
    if getattr(cfg.agent, "actor_type", "MLP") == "MLP_tanh":
        from torchrl.modules.models import MLP
        actor_nn = MLP(in_features=input_shape[0],
                       activation_class=torch.nn.Tanh,
                       out_features=output_shape,
                       depth=cfg.agent.hidden_sizes.__len__(),
                       num_cells=cfg.agent.hidden_sizes[-1],
                          )
        agent = create_actor_critic(
            input_shape,
            output_shape,
            actor_nn=actor_nn,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=cfg.agent.temperature,
            actor_depth=cfg.agent.hidden_sizes.__len__(),
            actor_cells=cfg.agent.hidden_sizes[-1],
        )
    if getattr(cfg.agent, "actor_type", "MLP") == "DSMNN":
        nn = DSMNN(N,
                    D,
                    latent_dim = 64,
                    deepset_width=16,
                    deepset_out_dim=16,
                    exp_unit_size= (64, 64),
                    relu_unit_size = (64, 64),
                    conf_unit_size = (64, 64),
                    )
        agent = create_actor_critic(
            input_shape,
            output_shape,
            actor_nn=nn,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=cfg.agent.temperature,

        )


    if getattr(cfg.agent, "actor_type", "MLP") == "MLP_independent":

        agent = create_independent_actor_critic(number_nodes=N,
                                                actor_input_dimension=D,
                                                actor_in_keys = ["Q", "Y", "lambda"],
                                                critic_in_keys=["observation"],
                                                action_spec = action_spec,
                                                temperature=cfg.agent.temperature,
                                                actor_depth=cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=cfg.agent.hidden_sizes[-1],
                                                type=1,
                                                network_type="MLP",)

    elif getattr(cfg.agent, "actor_type", "MLP") == "LMN_independent":
        agent = create_independent_actor_critic(number_nodes=N,
                                                actor_input_dimension=D,
                                                actor_in_keys=["Q", "Y", "lambda"],
                                                critic_in_keys=["observation"],
                                                action_spec=action_spec,
                                                temperature=cfg.agent.temperature,
                                                actor_depth=cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=cfg.agent.hidden_sizes[-1],
                                                type=1,
                                                network_type="LMN",)
    elif getattr(cfg.agent, "actor_type", "MLP") == "PMN_independent":
        agent = create_independent_actor_critic(number_nodes=N,
                                                actor_input_dimension=D,
                                                actor_in_keys=["Q", "Y", "lambda"],
                                                critic_in_keys=["observation"],
                                                action_spec=action_spec,
                                                temperature=cfg.agent.temperature,
                                                actor_depth=cfg.agent.hidden_sizes.__len__(),
                                                actor_cells=cfg.agent.hidden_sizes[-1],
                                                type=1,
                                                network_type="PMN",
                                                relu_max = getattr(cfg, "relu_max", 10),)

    elif getattr(cfg.agent, "actor_type", "MLP") == "PMN":
        mono_nn = PMN(input_size=input_shape[0],
                      output_size=output_shape,
                      hidden_sizes=cfg.agent.hidden_sizes,
                      relu_max=getattr(cfg, "relu_max", 0.1),
                      )
        agent = create_actor_critic(
            input_shape,
            output_shape,
            actor_nn=mono_nn,
            in_keys=["observation"],
            action_spec=action_spec,
            temperature=cfg.agent.temperature,
            actor_depth=cfg.agent.hidden_sizes.__len__(),
            actor_cells=cfg.agent.hidden_sizes[-1],
        )


    actor = agent.get_policy_operator().to(device)
    critic = agent.get_value_operator().to(device)

    adv_module = GAE(
        gamma=cfg.loss.gamma,
        lmbda=cfg.loss.gae_lambda,
        value_network=critic,
        average_gae=False,
    )
    # Create the collector

    make_env_funcs = [lambda i=i: training_env_generator.sample(true_ind = i) for i in training_env_generator.context_dicts.keys()]
    # Get lta for each environment
    training_env_info = {}
    for (e,i) in enumerate(training_env_generator.context_dicts.keys()):
        training_env_info[e] = {"lta": training_env_generator.context_dicts[i]["lta"],
                                "context_id": i}

    if "PMN" not in getattr(cfg.agent, "actor_type", "MLP") and "LMN" not in getattr(cfg.agent, "actor_type", "MLP"):
        collector = MultiSyncDataCollector(
            create_env_fn= make_env_funcs,
            policy=agent.get_policy_operator(),
            frames_per_batch=cfg.collector.frames_per_batch,
            total_frames=cfg.collector.total_frames,
            device=device,
            storing_device=device,
            env_device="cpu",
            max_frames_per_traj=cfg.collector.max_frames_per_traj,
            split_trajs=True,
            reset_when_done=True,

    )
    else:
        collector = MultiSyncDataCollector(
                create_env_fn= make_env_funcs,
                policy=agent.get_policy_operator(),
                frames_per_batch=cfg.collector.frames_per_batch,
                total_frames=cfg.collector.total_frames,
                device=device,
                storing_device=device,
                env_device="cpu",
                max_frames_per_traj=cfg.collector.max_frames_per_traj,
                split_trajs=True,
                reset_when_done=True,

        )

    # # create data buffer
    sampler = SamplerWithoutReplacement()
    data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.frames_per_batch),
        sampler=sampler,
        batch_size=cfg.loss.mini_batch_size,  # amount of samples to be sampled when sample is called
    )

    long_term_sampler = SamplerWithoutReplacement(shuffle=False)
    long_term_data_buffer = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(cfg.collector.map_interval),
        sampler=sampler,
        batch_size=cfg.collector.map_interval,  # amount of samples to be sampled when sample is called
    )

    ## Create PPO loss module
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=cfg.loss.clip_epsilon,
        entropy_coef=cfg.loss.entropy_coef,
        normalize_advantage=cfg.loss.norm_advantage,
        loss_critic_type="l2"
    )


    # Create the optimizer
    optimizer = torch.optim.Adam(
        loss_module.parameters(),
        lr= cfg.optim.lr,
        weight_decay=0,
    )

    # create wandb logger
    if logger is None:
        experiment_name = generate_exp_name(f"PPO_MLP", "Solo")
        logger = get_logger(
                "wandb",
                logger_name="..\\logs",
                experiment_name= experiment_name,
                wandb_kwargs={
                    "config": cfg.as_dict(),
                    "project": cfg.logger.project,
                },
            )
    #wandb.log({"init/init": True})
    #wandb.watch(q_module[0], log="parameters", log_freq=10)
    # wandb.watch(actor, log="all", log_freq=100, log_graph=False)

    # # Save the cfg as a yaml file and upload to wandb
    # with open(os.path.join(logger.experiment.dir, cfg.context_set), "w") as file:
    #     json.dump(make_serializable(training_env_generator.context_dicts), file)
    # wandb.save(cfg.context_set)


    # Main loop
    collected_frames = 0
    start_time = time.time()
    sampling_start = time.time()
    num_updates = cfg.loss.num_updates
    max_grad = cfg.optim.max_grad_norm
    pol_losses = torch.zeros(num_updates, device=device)
    mask_losses = torch.zeros(num_updates, device = device)
    num_mini_batches = cfg.collector.frames_per_batch // cfg.loss.mini_batch_size

    total_network_updates = (
            (cfg.collector.total_frames // cfg.collector.frames_per_batch) * cfg.loss.num_updates * num_mini_batches
    )
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, disable = disable_pbar)
    num_network_updates = 0
    # initialize the artifact saving params
    best_eval_backlog = np.inf
    artifact_name = logger.exp_name
    prev_log_info = {}
    for i, data in enumerate(collector):
        log_info = {}
        # Get and log evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            prev_map_frame = ((i - 1)) * cfg.collector.frames_per_batch // cfg.collector.map_interval
            cur_map_frame = (i) * cfg.collector.frames_per_batch // cfg.collector.map_interval
            final = collected_frames >= collector.total_frames
            # if (i >= 0 and (prev_map_frame < cur_map_frame)) or final:
            #     state_action_figure = create_action_maps2(actor, eval_env_generator.sample(), cfg,
            #                                              data_buffer=long_term_data_buffer,
            #                                               collected_frames=collected_frames,
            #                                              last_train_backlog = prev_log_info.get("train/avg_mean_normalized_backlog", None),
            #                                              last_eval_backlog = prev_log_info.get("eval_normalized/normalized_lta_backlog_training_envs", None))
            #     wandb.log(
            #         {f"StateActionMap/{collected_frames}": wandb.Image(state_action_figure), "step": collected_frames})
            prev_test_frame = ((i - 1) * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            cur_test_frame = (i * cfg.collector.frames_per_batch) // cfg.collector.test_interval
            if (i >= 1 and (prev_test_frame < cur_test_frame)) or final:
                actor.eval()
                eval_start = time.time()
                training_env_ids = list(training_env_generator.context_dicts.keys())
                eval_log_info, eval_tds = evaluate_dqn_agent(actor, eval_env_generator, training_env_ids, pbar, cfg,
                                                             device)

                eval_time = time.time() - eval_start
                log_info.update(eval_log_info)

                # Save the agent if the eval backlog is the best
                if eval_log_info["eval/lta_backlog_training_envs"] < best_eval_backlog:
                    best_eval_backlog = eval_log_info["eval/lta_backlog_training_envs"]
                    torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                    agent_artifact = wandb.Artifact(f"trained_actor_module_{artifact_name}", type="model")
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                    wandb.log_artifact(agent_artifact, aliases=["best", "latest"])
                else:
                    torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                    agent_artifact = wandb.Artifact(f"trained_actor_module.pt_{artifact_name}", type="model")
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
                    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
                    wandb.log_artifact(agent_artifact, aliases=["latest"])
                # log all of the state action figures to wandb

                actor.train()


        # print(f"Device of data: {data.device}")
        sampling_time = time.time() - sampling_start
        data = data[data["collector", "mask"]]

        pbar.update(data.numel())
        current_frames = data.numel()
        collected_frames += current_frames
        # drop data if [collector, mask] is False

        #env_datas = split_trajectories(data)
        # Get and log training rewards and episode lengths
        #combine all data that has the same context_id
        # data = data[data["collector", "mask"]]
        if data["logits"].dim() > 2:
            data["logits"] = data["logits"].squeeze()
        if data["action"].dim() > 2:
            data["action"] = data["action"].squeeze()
        for context_id in data["context_id"].unique():
            env_data = data[data["context_id"] == context_id]
            env_data = env_data[env_data["collector", "mask"]]

            # first check if all of the env_data is from the same environment
            # if not (env_data["collector", "traj_ids"] == env_data["collector", "traj_ids"][0]).all():
            #     raise ValueError("Data from multiple environments is being logged")
            context_id = env_data.get("context_id", None)
            if context_id is not None:
                context_id = context_id[0].item()
            baseline_lta = env_data.get("baseline_lta", None)
            if baseline_lta is not None:
                env_lta = baseline_lta[-1]
            mean_episode_reward = env_data["next", "reward"].mean()
            mean_backlog = env_data["next", "ta_mean"][-1]
            std_backlog = env_data["next", "ta_stdev"][-1]
            # mean_backlog = env_data["next", "backlog"].float().mean()
            normalized_backlog = mean_backlog / env_lta
            valid_action_fraction = (env_data["mask"] * env_data["action"]).sum().float() / env_data["mask"].shape[0]
            log_header = f"train/context_id_{context_id}"

            log_info.update({f'{log_header}/mean_episode_reward': mean_episode_reward.item(),
                             f'{log_header}/mean_backlog': mean_backlog.item(),
                             f'{log_header}/std_backlog': std_backlog.item(),
                             f'{log_header}/mean_normalized_backlog': normalized_backlog.item(),
                             f'{log_header}/valid_action_fraction': valid_action_fraction.item(), })



        # Get average mean_normalized_backlog across each context
        avg_mean_normalized_backlog = np.mean([log_info[f'train/context_id_{i}/mean_normalized_backlog'] for i in training_env_generator.context_dicts.keys()])
        log_info.update({"train/avg_mean_normalized_backlog": avg_mean_normalized_backlog})
        # compute the fraction of times the chosen action was invalid
        """
        data["mask"] is a binary tensor for each possible action, where False means the action was invalid
        data["action"] is the action chosen by the agent which is a one-hot binary tensor
        data["mask"] * data["action"] will be a binary tensor where the action was invalid
        """
        data = data[data["collector", "mask"]]

        # optimization steps
        training_start = time.time()
        losses = TensorDict({}, batch_size=[cfg.loss.num_updates, num_mini_batches])
        value_estimates = torch.zeros(num_updates, device=device)
        q_value_estimates = torch.zeros(num_updates, device=device)
        long_term_data_buffer.extend(data)
        for j in range(num_updates):
            # Compute GAE
            with torch.no_grad():
                data = adv_module(data.to(device, non_blocking=True))
                value_estimates[j] = data["state_value"].mean()
                q_value_estimates[j] = data["value_target"].mean()
            data_reshape = data.reshape(-1)
            # Update the data buffer
            data_buffer.extend(data_reshape)
            for k, batch in enumerate(data_buffer):
                # Linearly decrease the learning rate and clip epsilon
                alpha = 1.0
                if cfg.optim.anneal_lr:
                    alpha = 1 - (num_network_updates / total_network_updates)
                    for group in optimizer.param_groups:
                        group["lr"] = cfg.optim.lr * alpha

                num_network_updates += 1

                batch["sample_log_prob"] = batch["sample_log_prob"].squeeze()
                # batch["action"] = batch["action"].squeeze()
                # Get a data batch
                batch = batch.to(device, non_blocking=True)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, k] = loss.select(
                    "loss_critic", "loss_entropy", "loss_objective", "entropy", "ESS"
                )
                loss_sum = (
                        loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]
                )
                # Backward pass
                loss_sum.backward()

                ## Collect all gradient information if debugging
                # for name, param in loss_module.named_parameters():
                #     if param.grad is None:
                #         print(f"None gradient in actor {name}")
                #     else:
                #         print(f"{name} gradient: {param.grad.mean()}")
                torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=cfg.optim.max_grad_norm
                )

                # Update the networks
                optimizer.step()
                optimizer.zero_grad()
        pbar.set_description("Training")

        training_time = time.time() - training_start

        # Get and log q-values, loss, epsilon, sampling time and training time
        for key, value in loss.items():
            if key not in ["loss_critic", "loss_entropy", "loss_objective"]:
                log_info.update({f"train/{key}": value.mean().item()})
            else:
                log_info.update({f"train/{key}": value.sum().item()})
        log_info.update(
            {
                "train/lr": alpha * cfg.optim.lr,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
                "train/q_values": q_value_estimates.mean().item(),
                "train/value_estimate": value_estimates.mean().item(),
            }
        )



        try:
            #for key, value in log_info.items():
                #logger.log(key, value, collected_frames)
            log_info["trainer/step"] = collected_frames
            wandb.log(log_info, step=collected_frames)
        except Exception as e:
            print(e)
        prev_log_info = deepcopy(log_info)
        collector.update_policy_weights_()
        sampling_start = time.time()

    # Test Model last time
    # actor.eval()
    # eval_start = time.time()
    # training_env_ids = list(training_env_generator.context_dicts.keys())
    # eval_log_info, eval_tds = evaluate_dqn_agent(actor, eval_env_generator, training_env_ids, pbar, cfg,
    #                                              device)
    #
    # eval_time = time.time() - eval_start
    # log_info.update(eval_log_info)
    #
    # # Save the agent if the eval backlog is the best
    # if eval_log_info["eval/lta_backlog_training_envs"] < best_eval_backlog:
    #     best_eval_backlog = eval_log_info["eval/lta_backlog_training_envs"]
    #     torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
    #     agent_artifact = wandb.Artifact(f"trained_actor_module_{artifact_name}", type="model")
    #     agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
    #     agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
    #     wandb.log_artifact(agent_artifact, aliases=["best", "latest"])
    # else:
    torch.save(agent.state_dict(), os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
    agent_artifact = wandb.Artifact(f"trained_actor_module.pt_{artifact_name}", type="model")
    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"trained_actor_module.pt"))
    agent_artifact.add_file(os.path.join(logger.experiment.dir, f"config.yaml"))
    wandb.log_artifact(agent_artifact, aliases=["latest"])


