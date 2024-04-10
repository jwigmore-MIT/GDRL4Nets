import json
import os
from torchrl_development.utils.configuration import load_config
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic
import torch
from torchrl.record.loggers import get_logger
from datetime import datetime
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
from tqdm import tqdm
from torchrl_development.utils.metrics import compute_lta
import numpy as np
from torchrl_development.maxweight import MaxWeightActor
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torchrl_development.mdp_actors import MDP_actor, MDP_module
from MDP_Solver.SingleHopMDP import SingleHopMDP
import sys
from torchrl.data.replay_buffers import ReplayBuffer, ListStorage, LazyTensorStorage, SamplerWithoutReplacement
import tensordict
from tensordict import TensorDict

from experiments.experiment8.maxweight_comparison.CustomNNs import FeedForwardNN, LinearNetwork, MaxWeightNetwork, NN_Actor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))


def train_module2(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.CrossEntropyLoss(), weight_decay = 1e-5):
    loss_fn = loss_fn
    optimizer = Adam(module.parameters(), lr=lr, weight_decay=weight_decay)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    last_n_losses = []
    for epoch in pbar:
        # add learning rate decay
        alpha = 1 - (epoch / num_training_epochs)
        for group in optimizer.param_groups:
            group["lr"] = lr * alpha

        for mb, td in enumerate(replay_buffer):
            optimizer.zero_grad()
            td["Q"] = td["Q"].float()
            td["Y"] = td["Y"].float()
            A = module(td)
            loss = loss_fn(A['logits'].float(), td["target_action"].float())
            loss.backward(retain_graph = False)
            optimizer.step()
            if mb % 10 == 0:
                last_n_losses.append(loss.item())
                pbar.set_postfix({f"Epoch": epoch, 'mb': mb,  f"Loss": loss.detach().item()})
                if len(last_n_losses) > 10:
                    last_n_losses.pop(0)
                    if np.std(last_n_losses) < 1e-6:
                        break
        # stop training if loss converges

def get_module_error_rate(module, td, inputs = ["Q", "Y"]):
    module.eval()
    actions = module(torch.cat([td[key] for key in inputs], dim=1))
    error = torch.norm(actions - td["action"].float())
    error_rate = error / td["action"].shape[0]
    return error_rate, error




if __name__ == "__main__":
    import pickle
    env_id = 1
    rollout_length = 10000
    num_rollouts = 5
    training_epochs = 500
    learning_rate = 0.001

    pickle_string = f"id{env_id}_nr{num_rollouts}_rl{rollout_length}"

    new_mdp_trajectories = True

    actor_depth = 2
    actor_cells = 64

    results = {}
    test_context_set_path = 'SH1_context_set.json'
    mdp_path = "MDP_Solver/saved_mdps/4_9_pm_SH1/SH1_0_MDP.p"

    # Load all testing contexts
    context_set_path = 'SH1_context_set.json'
    test_context_set = json.load(open(context_set_path, 'rb'))

    # Create a generator from test_context_set
    make_env_parameters = {"observe_lambda": False,
                           "device": "cpu",
                           "terminal_backlog": 5000,
                           "inverse_reward": True,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": False,
                           "convergence_threshold": 0.1,
                           "terminate_on_lta_threshold": False, }

    env_generator_seed = 531997
    env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=env_generator_seed)

    base_env = env_generator.sample(env_id)
    env_generator.clear_history()

    input_shape = torch.Tensor([base_env.observation_spec["Q"].shape[0] + base_env.observation_spec["Y"].shape[0]]).int()
    output_shape = base_env.action_spec.space.n

    # Create mlp agent
    mlp_agent = create_actor_critic(input_shape=input_shape, output_shape=output_shape, in_keys=["observation"],
                                     action_spec=base_env.action_spec, temperature=0.1, actor_depth=actor_depth,
                                     actor_cells=actor_cells)
    mlp_agent.load_state_dict(torch.load("env1_mlp_model_2905000.pt"))

    # Create MW NN Agent
    mw_nn_agent = create_maxweight_actor_critic(input_shape=input_shape, output_shape=output_shape,
                                                action_spec=base_env.action_spec, in_keys=["Q", "Y"],
                                                init_weights= torch.Tensor([1,1]),
                                                temperature=10
                                                )
    # Create MaxWeight agent
    mw_agent = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    print("Collecting trajectories using agent and maxweight policy")

    # fresh MLP agent
    new_mlp_agent = create_actor_critic(input_shape=input_shape, output_shape=output_shape, in_keys=["observation"],
                                        action_spec=base_env.action_spec, temperature=0.1, actor_depth=actor_depth,
                                        actor_cells=actor_cells)

    # Collect trajectories from the MDP agent and the MaxWeight policy
    # or load them if they have already been collected
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        mlp_tds_path = f"mlp_tds_{pickle_string}.pkl"
        mw_tds_path = f"mw_tds_{pickle_string}.pkl"
        try:
            mlp_tds = pickle.load(open(mlp_tds_path, 'rb'))
            print("Loaded mlp_tds")
        except FileNotFoundError:
            mlp_tds = []
            for n in range(num_rollouts):
                env = env_generator.sample(env_id)
                print(f"Collecting trajectory {n} from MLP agent")
                mlp_td = env.rollout(policy=mlp_agent, max_steps=rollout_length)
                mlp_tds.append(mlp_td)
            pickle.dump(mlp_tds, open(mlp_tds_path, 'wb'))
            env_generator.reseed()

        try:
            mw_tds = pickle.load(open(mw_tds_path, 'rb'))
            print("Loaded mw_tds")
        except FileNotFoundError:
            mw_tds = []
            for n in range(num_rollouts):
                env = env_generator.sample(env_id)
                print(f"Collecting trajectory {n} from MaxWeight policy")
                mw_td = env.rollout(policy=mw_agent, max_steps=rollout_length)
                mw_tds.append(mw_td)
            pickle.dump(mw_tds, open(mw_tds_path, 'wb'))
            env_generator.reseed()

    # Compute the mean lta for the mdp agent and mw agent's trajectories
    mlp_ltas = [compute_lta(td["backlog"]) for td in mlp_tds]
    mlp_mean_lta = np.mean(mlp_ltas, axis=0)
    mlp_std_lta = np.std(mlp_ltas, axis=0)

    mw_ltas = [compute_lta(mw_td["backlog"]) for mw_td in mw_tds]
    mw_mean_lta = np.mean(mw_ltas, axis=0)
    mw_std_lta = np.std(mw_ltas, axis=0)


    # plot the lta of the mdp agent and the maxweight policy
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.plot(mlp_mean_lta, label="MLP Agent")
    ax.fill_between(range(len(mlp_mean_lta)), mlp_mean_lta - mlp_std_lta, mlp_mean_lta + mlp_std_lta, alpha=0.5)
    ax.plot(mw_mean_lta, label="MaxWeight")
    ax.fill_between(range(len(mw_mean_lta)), mw_mean_lta - mw_std_lta, mw_mean_lta + mw_std_lta, alpha=0.5)
    ax.set_ylim(0, mw_mean_lta.max() * 2)
    ax.legend()
    ax.set_title("LTA Comparison between MLP Agent and MaxWeight Policy")
    fig.show()

    # Create the training dataset by:
    #   Merging all the mlp_trajectories in tds into a single tensor dict
    #   modifying the td to have a "true_action" key
    # Merge all tensordicts in tds into a single tensordict
    td = torch.cat(mlp_tds)
    td["target_action"] = td["action"].clone().long()
    td["target_logits"] = td["logits"].detach().clone()

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size = td.shape[0]),
                                 batch_size = int(td.shape[0]/10),
                                 sampler = SamplerWithoutReplacement(shuffle=True))
    replay_buffer.extend(td)
    sample = replay_buffer.sample()



    # Train the mw_nn agent
    train_module2(mw_nn_agent, replay_buffer, num_training_epochs=training_epochs, lr=learning_rate,
                  loss_fn=nn.BCEWithLogitsLoss(), weight_decay=0)
    # generator a trajectory from the MLP agent agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        mw_nn_tds = []
        for n in range(num_rollouts):
            print(f"Collecting trajectory {n} from MW NN agent")
            env = env_generator.sample(env_id)
            mw_nn_td = env.rollout(policy=mw_nn_agent, max_steps=rollout_length)
            mw_nn_tds.append(mw_nn_td)
        env_generator.reseed()


    # compute the mean lta for the mlp agent's trajectories
    mw_nn_ltas = [compute_lta(mlp_nn_td["backlog"]) for mlp_nn_td in mw_nn_tds]
    mw_nn_mean_lta = np.mean(mw_nn_ltas, axis=0)
    mw_nn_std_lta = np.std(mw_nn_ltas, axis=0)

    # train and test the new mlp agent
    train_module2(new_mlp_agent, replay_buffer, num_training_epochs=training_epochs, lr=learning_rate,
                    loss_fn=nn.BCEWithLogitsLoss(), weight_decay=0)
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        new_mlp_tds = []
        for n in range(num_rollouts):
            print(f"Collecting trajectory {n} from new MLP agent")
            env = env_generator.sample(env_id)
            new_mlp_td = env.rollout(policy=new_mlp_agent, max_steps=rollout_length)
            new_mlp_tds.append(new_mlp_td)
        env_generator.reseed()

    # compute the mean lta for the mlp agent's trajectories
    new_mlp_ltas = [compute_lta(new_mlp_td["backlog"]) for new_mlp_td in new_mlp_tds]
    new_mlp_mean_lta = np.mean(new_mlp_ltas, axis=0)
    new_mlp_std_lta = np.std(new_mlp_ltas, axis=0)




    # Plot the LTA for all agents
    arrival_rates = env.base_env.arrival_rates
    # mw_nn_lta = compute_lta(mw_nn_td["backlog"])
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    # plot the arrival rates as a bar chart
    ax[0].bar(range(1, len(arrival_rates) + 1), arrival_rates)
    ax[0].set_title("Arrival Rates")
    ax[0].set_xlabel("Node")
    # Plot LTA for all trajectories
    ax[1].plot(mlp_mean_lta, label="MLP Agent")
    ax[1].fill_between(range(len(mlp_mean_lta)), mlp_mean_lta - mlp_std_lta, mlp_mean_lta + mlp_std_lta, alpha=0.2)
    ax[1].plot(mw_mean_lta, label="MaxWeight", color = 'red')
    ax[1].fill_between(range(len(mw_mean_lta)), mw_mean_lta - mw_std_lta, mw_mean_lta + mw_std_lta, alpha=0.2, color = 'red')
    ax[1].plot(mw_nn_mean_lta, label=f"MW  NN Agent")
    ax[1].fill_between(range(len(mw_nn_mean_lta)), mw_nn_mean_lta - mw_nn_std_lta, mw_nn_mean_lta + mw_nn_std_lta, alpha=0.2)
    ax[1].plot(new_mlp_mean_lta, label=f"New MLP Agent")
    ax[1].fill_between(range(len(new_mlp_mean_lta)), new_mlp_mean_lta - new_mlp_std_lta, new_mlp_mean_lta + new_mlp_std_lta, alpha=0.2)
    ax[1].legend()
    ax[1].set_title("Agent performance Comparison")
    arrival_rates_formatted = [float(f"{x:.2f}") for x in arrival_rates]
    ax[2].plot(mlp_mean_lta / mw_mean_lta, label="MLP Agent")
    ax[2].plot(mw_nn_mean_lta / mw_mean_lta, label="MW NN Agent")
    ax[2].plot(new_mlp_mean_lta / mw_mean_lta, label="New MLP Agent")
    # add text to the plot that shows the final normalized performance for each agent
    ax[2].text(len(mlp_mean_lta) + 200, mlp_mean_lta[-1] / mw_mean_lta[-1], f"{mlp_mean_lta[-1] / mw_mean_lta[-1]:.4f}")
    ax[2].text(len(mw_nn_mean_lta) + 200, mw_nn_mean_lta[-1] / mw_mean_lta[-1],
            f"{mw_nn_mean_lta[-1] / mw_mean_lta[-1]:.4f}")
    ax[2].text(len(new_mlp_mean_lta) + 200, new_mlp_mean_lta[-1] / mw_mean_lta[-1],f"{new_mlp_mean_lta[-1] / mw_mean_lta[-1]:.4f}")
    ax[2].set_xlim(0, len(mlp_mean_lta) + 1000)
    ax[2].set_title("Agent normalized performance Comparison")
    fig.suptitle(f"Imitation Learning LTA Comparison {env_id}")
    fig.tight_layout()
    fig.show()



    #