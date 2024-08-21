import json
import os
from torchrl_development.utils.configuration import load_config
from torchrl_development.envs.env_generators import EnvGenerator
from torchrl_development.actors import create_actor_critic
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
from torchrl_development.actors import create_actor_critic, create_maxweight_actor_critic, create_gnn_maxweight_actor_critic

from experiments.experiment8.maxweight_comparison.CustomNNs import FeedForwardNN, LinearNetwork, MaxWeightNetwork, \
    NN_Actor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))

def train_module2(module, replay_buffer, in_keys = ["Q", "Y"], num_training_epochs=5, lr=0.0001,
                 loss_fn = nn.BCEWithLogitsLoss(), weight_decay = 1e-5):
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
            td = module(td)
            loss = loss_fn(td['logits'], td["target_action"])
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


def get_module_error_rate(module, td, inputs=["Q", "Y"]):
    module.eval()
    actions = module(torch.cat([td[key] for key in inputs], dim=1))
    error = torch.norm(actions - td["action"].float())
    error_rate = error / td["action"].shape[0]
    return error_rate, error




if __name__ == "__main__":
    import pickle

    # Context id
    env_id = 0

    # Data generation parameters
    context_set_path = 'SH3_context_set_100_03251626.json'
    rollout_length = 10000
    num_rollouts = 3
    env_generator_seed = 531997

    # Training parameters
    training_epochs= 300
    learning_rate = 0.001

    # Data Saving
    pickle_string = f"id{env_id}_nr{num_rollouts}_rl{rollout_length}"

    # MLP actor parameters
    model_path = 'SH3_trained_agents/MLP_SH3a_model_2905000.pt'
    actor_depth = 2
    actor_cells = 64

    # To store results
    results = {}

    # Load all testing contexts
    context_set = json.load(open(context_set_path, 'rb'))

    # Create a generator from test_context_set
    make_env_parameters = {"observe_lambda": False,
                           "device": "cpu",
                           "terminal_backlog": 5000,
                           "inverse_reward": True,
                           "stat_window_size": 100000,
                           "terminate_on_convergence": False,
                           "convergence_threshold": 0.1,
                           "terminate_on_lta_threshold": False, }

    env_generator = EnvGenerator(context_set, make_env_parameters, env_generator_seed=env_generator_seed)

    base_env = env_generator.sample(env_id)
    env_generator.clear_history()

    # Get the input and output shapes
    input_shape = torch.Tensor(
        [base_env.observation_spec["Q"].shape[0] + base_env.observation_spec["Y"].shape[0]]).int()
    output_shape = base_env.action_spec.space.n

    # Create MLP agent
    mlp_agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        actor_depth=2,
        actor_cells=64
    )


    # Load agent
    mlp_agent.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Get MaxWeight lta backlog from context_set
    mw_mean_lta = np.array(context_set['context_dicts'][str(env_id)]["lta"])


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

    # Compute the mean lta for the mdp agent and mw agent's trajectories
    mlp_ltas = [compute_lta(td["backlog"]) for td in mlp_tds]
    mlp_mean_lta = np.mean(mlp_ltas, axis=0)
    mlp_std_lta = np.std(mlp_ltas, axis=0)


    # merge tds
    td = torch.cat(mlp_tds)

    # Create the training dataset by modifying the td to have a "true_action" key
    # the true actions should also convert the one hot encoding to a single integer
    td["target_action"] = td["action"].argmax(dim=1).long()
    td["target_logits"] = td["logits"].detach().clone()

    # Create Replay Buffer
    replay_buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=td.shape[0]),
                                 batch_size=int(td.shape[0] / 10),
                                 sampler=SamplerWithoutReplacement(shuffle=True))
    replay_buffer.extend(td)
    sample = replay_buffer.sample()


    # Create MaxWeight NN
    mwn_agent = create_maxweight_actor_critic(input_shape=input_shape, output_shape=output_shape,
                                                action_spec=base_env.action_spec, in_keys=["Q", "Y"],
                                                temperature=10
                                                )

    # Train the mw_nn agent
    train_module2(mwn_agent, replay_buffer, num_training_epochs=training_epochs, lr=learning_rate,
                  loss_fn=nn.CrossEntropyLoss(), weight_decay=0)
    # Compute error on the training data
    mwn_agent.eval()
    # mw_nn_error_rate, mw_nn_error = get_module_error_rate(mwn_agent, td)

    # Collect the learned weights
    w = mwn_agent.mwn_weights().detach().squeeze().numpy()[1:]
    w_normalized = w / np.sum(w)

    # Create a MW_NN actor

    # generator a trajectory from the MW_NN agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        mw_nn_tds = []
        for n in range(num_rollouts):
            print(f"Collecting trajectory {n} from trained MW_NN agent")
            env = env_generator.sample(env_id)
            mw_nn_td = env.rollout(policy=mwn_agent, max_steps=rollout_length)
            mw_nn_tds.append(mw_nn_td)

    mw_nn_ltas = [compute_lta(mw_nn_td["backlog"]) for mw_nn_td in mw_nn_tds]
    mw_nn_mean_lta = np.mean(mw_nn_ltas, axis=0)
    mw_nn_std_lta = np.std(mw_nn_ltas, axis=0)

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

    ax[1].plot(mw_nn_mean_lta, label=f"MW  NN Agent")
    ax[1].fill_between(range(len(mw_nn_mean_lta)), mw_nn_mean_lta - mw_nn_std_lta, mw_nn_mean_lta + mw_nn_std_lta,
                       alpha=0.2)

    ax[1].legend()
    ax[1].set_title("Agent performance Comparison")
    arrival_rates_formatted = [float(f"{x:.2f}") for x in arrival_rates]
    ax[2].plot(mlp_mean_lta / mw_mean_lta, label="MLP Agent")
    ax[2].plot(mw_nn_mean_lta / mw_mean_lta, label="MW NN Agent")

    # add text to the plot that shows the final normalized performance for each agent
    ax[2].text(len(mlp_mean_lta) + 200, mlp_mean_lta[-1] / mw_mean_lta, f"{mlp_mean_lta[-1] / mw_mean_lta:.4f}")
    ax[2].text(len(mw_nn_mean_lta) + 200, mw_nn_mean_lta[-1] / mw_mean_lta,
               f"{mw_nn_mean_lta[-1] / mw_mean_lta:.4f}")

    ax[2].set_xlim(0, len(mlp_mean_lta) + 1000)
    ax[2].set_title("Agent normalized performance Comparison")
    fig.suptitle(f"Imitation Learning LTA Comparison {env_id}")
    fig.tight_layout()
    fig.show()


# Create state-action maps for each agent
mwn_sa_map = {}
mlp_sa_map = {}

for n, td in enumerate(mw_nn_tds):
    for Q, Y, action in zip(td["Q"], td["Y"], td["action"]):
        state = tuple(Q.tolist() + Y.tolist())
        if action[0] == 1:  # skip all idling actions
            continue
        if state not in mwn_sa_map:
            mwn_sa_map[state] = []
        mwn_sa_map[state].append(action)

for n, td in enumerate(mlp_tds):
    for Q, Y, action in zip(td["Q"], td["Y"], td["action"]):
        state = tuple(Q.tolist() + Y.tolist())
        if action[0] == 1:  # skip all idling actions
            continue
        if state not in mlp_sa_map:
            mlp_sa_map[state] = []
        mlp_sa_map[state].append(action)

# Find where MWN and MLP agent differ
diffs = {}
for state, actions in mlp_sa_map.items():
    if state not in mwn_sa_map:
        continue
    if not (actions[0] == mwn_sa_map[state][0]).all().item():
        diffs[state] = (actions[0].numpy(), mwn_sa_map[state][0].numpy())  # (MLP, MWN)

# print table
from tabulate import tabulate
# Prepare the headers for the table
headers = ["Q", "Y", "W*Q*Y", "MWN Action", "MLP Action"]

# Prepare the data for the table
table_data = []
for state, (mlp_action, mwn_action) in diffs.items():
    Q = state[:5]
    Y = state[5:]
    # compute elemtwise product of Q, Y and W
    WQY = np.round(w_normalized*Q*Y, 2)
    table_data.append([Q, Y, WQY, mwn_action[1:], mlp_action[1:]])

# Print the table
print(tabulate(table_data, headers, tablefmt="pretty"))

import matplotlib.pyplot as plt
import numpy as np

# Now lets look at a histogram of the Q and Y values in the tds generated by
# the MLP policy
Qs_mlp = []
Ys_mlp = []
for td in mlp_tds:
    Qs_mlp.extend(td["Q"][td["action"].argmax(dim = 1)].tolist())
    Ys_mlp.extend(td["Y"][td["action"].argmax(dim = 1)].tolist())

Qs_mlp = np.array(Qs_mlp)
Ys_mlp = np.array(Ys_mlp)

# Now lets look at a histogram of the Q and Y values in
# the difference table
Qs_diff = []
Ys_diff = []

for state, (mlp_action, mwn_action) in diffs.items():
    Qs_diff.append(state[:5][mwn_action[1:].argmax()])
    Ys_diff.append(state[5:][mwn_action[1:].argmax()])
Qs_diff = np.array(Qs_diff)
Ys_diff = np.array(Ys_diff)

# Create the plots
fig, ax = plt.subplots(2, 1, figsize=(10, 10))

# Plot Q values
ax[0].hist(Qs_mlp.flatten(), bins=100, alpha=0.5, label='MLP', density=True)
ax[0].hist(Qs_diff.flatten(), bins=100, alpha=0.5, label='Diff', density=True)
ax[0].set_title("Q values")
ax[0].legend()

# Plot Y values
ax[1].hist(Ys_mlp.flatten(), bins=100, alpha=0.5, label='MLP', density=True)
ax[1].hist(Ys_diff.flatten(), bins=100, alpha=0.5, label='Diff', density=True)
ax[1].set_title("Y values")
ax[1].legend()

fig.show()
