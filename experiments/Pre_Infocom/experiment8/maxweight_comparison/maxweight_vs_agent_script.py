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

from experiments.experiment8.maxweight_comparison.CustomNNs import FeedForwardNN, LinearNetwork, MaxWeightNetwork, NN_Actor


def max_weight_policy(Q,Y, w = None):
    """Computes the MaxWeight policy action given the Q and Y array"""
    A = torch.zeros((Q.shape[0],Q.shape[1]+1), dtype=torch.int)
    if w is None:
        w = torch.ones(Q.shape[1])
    for i in range(Q.shape[0]):
        v = Q[i]*Y[i]*w
        if torch.all(v==0):
            A[i,0] = 1
        else:
            max_index = torch.argmax(v)
            A[i,max_index+1] = 1
    return A

def train_module(module, td, in_keys = ["Q", "Y"], num_training_epochs=1000, lr=0.001,
                 loss_fn = nn.BCEWithLogitsLoss()):
    loss_fn = loss_fn
    optimizer = Adam(module.parameters(), lr=lr)
    pbar = tqdm(range(num_training_epochs), desc=f"Training {module.__class__.__name__}")
    last_n_losses = []
    for epoch in pbar:
        optimizer.zero_grad()
        x = torch.cat([td[key] for key in in_keys], dim=1)
        # Q = td["Q"].clone().float().detach().requires_grad_(True)
        # Y = td["Y"].clone().float().detach().requires_grad_(True)
        A = module(x)
        loss = loss_fn(A.float(), td["action"].float())
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            last_n_losses.append(loss.item())
            pbar.set_postfix({f"Epoch": epoch, f"Loss": loss.item()})
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
    env_id = 30
    rollout_length = 10000
    training_epochs = 10000

    print(f"Running experiment8_test_agent.py for env_id {env_id}")
    SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
    CONFIG_PATH = os.path.join(SCRIPT_PATH, "experiment8_model_test.yaml")

    experiment_name = "experiment8_model_test"

    # Load all testing contexts
    test_context_set = json.load(open('SH2u_context_set_100_03211523.json'))
    # Create a generator from test_context_set
    make_env_parameters = {"observe_lambda": True,
                            "device": "cpu",
                            "terminal_backlog": 5000,
                            "inverse_reward": True,
                            "stat_window_size": 100000,
                            "terminate_on_convergence": False,
                            "convergence_threshold": 0.1,
                            "terminate_on_lta_threshold": False,}

    env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=0)

    base_env = env_generator.sample(10)
    env_generator.clear_history()

    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n

    # Create agent
    agent = create_actor_critic(
        input_shape,
        output_shape,
        in_keys=["observation"],
        action_spec=base_env.action_spec,
        temperature=0.1,
        )

    # Set device
    device = "cpu"
    # count the number of parameters in the agent
    num_params = sum(p.numel() for p in agent.get_policy_operator().parameters())
    # Load agent
    agent.load_state_dict(torch.load('model_2905000.pt', map_location=device))
    mw_agent = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    print("Collecting trajectories using agent and maxweight policy")
    # generator a trajectory from the agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        env = env_generator.sample(env_id)
        env.reset()
        td = env.rollout(policy=agent, max_steps = rollout_length)

        env.reset()
        mw_td = env.rollout(policy=mw_agent, max_steps = rollout_length)


    # compare the lta of the two trajectories
    agenta_lta = compute_lta(td["backlog"])
    mw_lta = compute_lta(mw_td["backlog"])

    # plot the lta of the two trajectories
    plt.plot(agenta_lta, label="Agent A")
    plt.plot(mw_lta, label="MaxWeight")
    plt.legend()
    plt.show()

    # compare the action frequencies of the two trajectories
    agent_actions = pd.DataFrame(td["action"]).value_counts(normalize=True)
    mw_actions = pd.DataFrame(mw_td["action"]).value_counts(normalize=True)

    # plot the action frequencies of the two trajectories
    fig, ax = plt.subplots(1, 2)
    agent_actions.plot(kind='bar', ax=ax[0])
    mw_actions.plot(kind='bar', ax=ax[1])
    ax[0].set_title("Agent Action Frequencies")
    ax[1].set_title("MaxWeight Action Frequencies")
    plt.tight_layout()
    plt.show()


    # enumerate all the observations in the two trajectories
    agent_observations = pd.DataFrame(td["observation"])
    mw_observations = pd.DataFrame(mw_td["observation"])
    observation = pd.concat([agent_observations, mw_observations], axis=1)

    # create a dataframe with columns correspond to each element of 'Q', 'Y', and 'action'
    # for the td
    Q_df = pd.DataFrame(td["Q"].numpy(), columns=[f"Q{i+1}" for i in range(td["Q"].shape[1])])
    Y_df = pd.DataFrame(td["Y"].numpy(), columns=[f"Y{i+1}" for i in range(td["Y"].shape[1])])
    action_df = pd.DataFrame(td["action"].numpy(), columns=[f"A{i}" for i in range(td["action"].shape[1])])
    mw_actions = max_weight_policy(td["Q"], td["Y"])
    mw_action_df = pd.DataFrame(mw_actions.numpy(), columns=[f"Aw{i}" for i in range(td["action"].shape[1])])

    MW_NN = MaxWeightNetwork(td["Q"].shape[1])

    MW_NN_actions = MW_NN(torch.cat([td["Q"], td["Y"]], dim=1))

    # check if mw_action is equal to MW_NN_actions
    MW_comparison = torch.all(mw_actions == MW_NN_actions.detach().int())
    # Find where they are not equal
    MW_comparison_indices = torch.where(mw_actions != MW_NN_actions.detach().int())

    # create a dataframe comparing mw_actions and MW_NN_actions
    mw_nn_df = pd.DataFrame(MW_NN_actions.detach().numpy(), columns=[f"NN{i}" for i in range(td["action"].shape[1])])
    mw_df = pd.concat([mw_action_df, mw_nn_df], axis=1)


    td_df = pd.concat([Q_df, Y_df, action_df, mw_action_df ], axis=1)

    ## Can we learn weights w so that max_weight_policy(Q,Y,w) = agent_policy(Q,Y)?

    ''' Can we encode the maxweight policy as a neural network?
    Let z[i] be the ith output of the neural network. 
    Let Q[i] be the ith Q inputs to the neural network
    Let Y[i] be the ith Y inputs to the neural network
    Let w[i] be the ith weight of the neural network. There are only Q.shape[1] weights
    z[i] = Y[i]*Q[i]*w[i]
    A[i] = argmax(z[i]) + 1
    Create this neural network:
    '''

    '''
    Now train the MW_NN agent to match the agents policy
    '''
    # create a dataset

    train_module(MW_NN, td, num_training_epochs=training_epochs, lr=0.001, loss_fn = nn.BCELoss())

    # Run MW_NN on the trajectory
    MW_NN.eval()
    mw_nn_error_rate, mw_nn_error = get_module_error_rate(MW_NN, td)

    MW2_NN = MaxWeightNetwork(td["Q"].shape[1])
    train_module(MW2_NN, mw_td, in_keys=["Q", "Y"], num_training_epochs=training_epochs, lr=0.001, loss_fn=nn.BCELoss())

    MW2_NN.eval()
    mw2_nn_error_rate, mw2_nn_error = get_module_error_rate(MW2_NN, mw_td, inputs=["Q", "Y"])



    Linear_NN = LinearNetwork(td["Q"].shape[1]+td["Y"].shape[1], td["action"].shape[1])
    train_module(Linear_NN, td, num_training_epochs=training_epochs)

    # run Linear_NN on the trajectory
    Linear_NN.eval()
    linear_nn_error_rate, linear_nn_error = get_module_error_rate(Linear_NN, td)


    FF_NN = FeedForwardNN(td["Q"].shape[1]+td["Y"].shape[1], 32, td["action"].shape[1])
    train_module(FF_NN, td, num_training_epochs=training_epochs)

    # run FF_NN on the trajectory
    FF_NN.eval()
    ff_nn_error_rate, ff_nn_error = get_module_error_rate(FF_NN, td)


    # Feedforward NN with only using Q as input
    FF_NN_Q = FeedForwardNN(td["Q"].shape[1], 32, td["action"].shape[1])
    train_module(FF_NN_Q, td, in_keys=["Q"], num_training_epochs=training_epochs)

    FF_NN_Q.eval()
    ff_nn_q_error_rate, ff_nn_q_error = get_module_error_rate(FF_NN_Q, td, inputs=["Q"])

    # Feedforward NN with only using Y as input
    FF_NN_Y = FeedForwardNN(td["Y"].shape[1], 32, td["action"].shape[1])
    train_module(FF_NN_Y, td, in_keys=["Y"], num_training_epochs=training_epochs)

    FF_NN_Y.eval()
    ff_nn_y_error_rate, ff_nn_y_error = get_module_error_rate(FF_NN_Y, td, inputs=["Y"])






    # compute error between mw_actions and agent
    mw_error = torch.norm(mw_actions - td["action"].float())
    mw_error_rate = mw_error / td["action"].shape[0]

    w = MW_NN.weights.detach().numpy()

    w_normalized = w / np.sum(w)

    Y_means = td["Y"].float().mean(dim=0).numpy()

    Y_means_normalized = Y_means / np.sum(Y_means)
    inv_Y_means_normalized = (1-Y_means_normalized)/(1-Y_means_normalized).sum()

    error = np.linalg.norm(w_normalized - inv_Y_means_normalized)

    # should be inversely proportional
    # what is the relationship between w and Y_means?

    arrival_rates = env.base_env.arrival_rates

    print(f"Arrival Rates {arrival_rates}")
    print(f"w = {w}")
    print(f"w_normalized = {w_normalized}")
    print(f"Y_means = {Y_means}")
    print(f"Y_means_normalized = {Y_means_normalized}")
    print(f"inv_Y_means_normalized = {inv_Y_means_normalized}")
    print(f"MW error = {mw_error}")
    print(f"MW error rate= {mw_error_rate}")
    print(f"NN error = {mw_nn_error}")
    print(f"NN error rate= {mw_nn_error_rate}")
    print(f"Linear NN error = {linear_nn_error}")
    print(f"Linear NN error rate= {linear_nn_error_rate}")
    print("FeedForward NN error = ", ff_nn_error)
    print("FeedForward NN error rate = ", ff_nn_error_rate)
    print("FeedForward NN Q error = ", ff_nn_q_error)
    print("FeedForward NN Q error rate = ", ff_nn_q_error_rate)
    print("FeedForward NN Y error = ", ff_nn_y_error)
    print("FeedForward NN Y error rate = ", ff_nn_y_error_rate)
    print(f"Max Weight NN weights = {MW_NN.weights}")
    print(f"Max Weight NN weights normalized = {w_normalized}")


    """
    The agent is clearly not learning a simple max-weight like policy 
    """


    MW_NN_agent = NN_Actor(module=MW_NN, in_keys=["Q", "Y"], out_keys=["action"])
    Linear_NN_agent = NN_Actor(module=Linear_NN, in_keys=["Q", "Y"], out_keys=["action"])
    FF_NN_agent = NN_Actor(module=FF_NN, in_keys=["Q", "Y"], out_keys=["action"])
    FF_NN_Q_agent = NN_Actor(module=FF_NN_Q, in_keys=["Q"], out_keys=["action"])
    FF_NN_Y_agent = NN_Actor(module=FF_NN_Y, in_keys=["Y"], out_keys=["action"])

    # generator a trajectory from the agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        env = env_generator.sample(env_id)
        env.reset()
        mw_nn_td = env.rollout(policy=MW_NN_agent, max_steps = rollout_length)
        env.reset()
        linear_nn_td = env.rollout(policy=Linear_NN_agent, max_steps = rollout_length)
        env.reset()
        ff_nn_td = env.rollout(policy=FF_NN_agent, max_steps = rollout_length)
        env.reset()
        ff_nn_q_td = env.rollout(policy=FF_NN_Q_agent, max_steps = rollout_length)
        env.reset()
        ff_nn_y_td = env.rollout(policy=FF_NN_Y_agent, max_steps = rollout_length)


    # compare the lta of all trajectories
    mw_nn_lta = compute_lta(mw_nn_td["backlog"])
    linear_nn_lta = compute_lta(linear_nn_td["backlog"])
    ff_nn_lta = compute_lta(ff_nn_td["backlog"])
    ff_nn_q_lta = compute_lta(ff_nn_q_td["backlog"])
    ff_nn_y_lta = compute_lta(ff_nn_y_td["backlog"])


    # plot the lta of the all trajectories
    plt.plot(agenta_lta, label="Agent A")
    plt.plot(mw_lta, label="MaxWeight")
    plt.plot(mw_nn_lta, label="MW_NN")
    plt.plot(linear_nn_lta, label="Linear_NN")
    plt.plot(ff_nn_lta, label="FF_NN")
    plt.plot(ff_nn_q_lta, label="FF_NN_Q")
    plt.plot(ff_nn_y_lta, label="FF_NN_Y")
    plt.ylim(0, 100)

    plt.legend()
    plt.show()

