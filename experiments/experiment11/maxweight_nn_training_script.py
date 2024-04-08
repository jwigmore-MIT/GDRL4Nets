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


from experiments.experiment8.maxweight_comparison.CustomNNs import FeedForwardNN, LinearNetwork, MaxWeightNetwork, NN_Actor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(PROJECT_DIR, 'MDP_Solver'))

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


def maxweight_nn_training_test(env_id=31, context_set_path = None, mdp_path = None, rollout_length=10000, training_epochs=10000):

    if context_set_path is None:
        context_set_path = 'SH1_context_set.json'


    # Load all testing contexts
    test_context_set = json.load(open(context_set_path, 'rb'))

    # Create a generator from test_context_set
    make_env_parameters = {"observe_lambda": True,
                            "device": "cpu",
                            "terminal_backlog": 5000,
                            "inverse_reward": True,
                            "stat_window_size": 100000,
                            "terminate_on_convergence": False,
                            "convergence_threshold": 0.1,
                            "terminate_on_lta_threshold": True,}

    env_generator = EnvGenerator(test_context_set, make_env_parameters, env_generator_seed=0)

    base_env = env_generator.sample(env_id)
    env_generator.clear_history()

    input_shape = base_env.observation_spec["observation"].shape
    output_shape = base_env.action_spec.space.n

    # Create agent
    mdp = pickle.load(open(os.path.join(PROJECT_DIR, mdp_path), 'rb'))
    mdp_module = MDP_module(mdp)
    agent = MDP_actor(mdp_module)

    # Set device
    device = "cpu"

    # Load agent
    # agent.load_state_dict(torch.load(model_path, map_location=device))
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

    # plot the lta of the all trajectories
    fig, ax = plt.subplots(1,1, figsize=(10, 10))
    ax.plot(agenta_lta, label="MDP Agent")
    ax.plot(mw_lta, label="MaxWeight")
    ax.set_ylim(0, mw_lta.max()*1.1)
    ax.legend()
    ax.set_title("MDP Agent Performance")
    fig.show()


    # compare the action frequencies of the two trajectories
    agent_actions = pd.DataFrame(td["action"]).value_counts(normalize=True)
    mw_actions = pd.DataFrame(mw_td["action"]).value_counts(normalize=True)




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


    print(f"MaxWeight error = {mw_error}")
    print(f"MaxWeight error rate= {mw_error_rate}")
    print(f"MaxWeight NN error = {mw_nn_error}")
    print(f"MaxWeight NN error rate= {mw_nn_error_rate}")
    print(f"Arrival Rates {arrival_rates}")
    print(f"w = {w}")
    print(f"w_normalized = {w_normalized}")
    print(f"Y_means = {Y_means}")
    print(f"Y_means_normalized = {Y_means_normalized}")
    print(f"inv_Y_means_normalized = {inv_Y_means_normalized}")


    """
    The agent is clearly not learning a simple max-weight like policy 
    """


    MW_NN_agent = NN_Actor(module=MW_NN, in_keys=["Q", "Y"], out_keys=["action"])

    # generator a trajectory from the agent
    with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
        env = env_generator.sample(env_id)
        env.reset()
        mw_nn_td = env.rollout(policy=MW_NN_agent, max_steps = rollout_length)
        env.reset()


    # compare the lta of all trajectories
    mw_nn_lta = compute_lta(mw_nn_td["backlog"])


    # plot the lta of the all trajectories
    fig, ax = plt.subplots(2,1, figsize=(10, 10))
    # plot the arrival rates as a bar chart
    ax[0].bar(range(1,len(arrival_rates)+1), arrival_rates)
    ax[0].set_title("Arrival Rates")
    ax[0].set_xlabel("Node")
    # Plot LTA for all trajectories
    ax[1].plot(agenta_lta, label="MDP Agent")
    ax[1].plot(mw_lta, label="MaxWeight")
    ax[1].plot(mw_nn_lta, label=f"MW_NN")
    ax[1].set_ylim(0, mw_lta.max()*1.1)
    ax[1].legend()
    ax[1].set_title("LTA")
    arrival_rates_formatted = [float(f"{x:.2f}") for x in arrival_rates]
    mw_nn_error_rate_formatted = f"{mw_nn_error_rate:.4f}"
    w_normalized = [float(f"{x:.4f}") for x in w_normalized]

    fig.suptitle(f"MW NN training comparison for env_id {env_id}")

    text_str = f"""
    env_id = {env_id}
    arrival_rates = {arrival_rates_formatted}
    mw_nn_error_rate = {mw_nn_error_rate_formatted}
    MW NN Weights = {w}
    Normalized Weights = {w_normalized}
    """
    plt.subplots_adjust(hspace=0.5, bottom=0.2)

    for i, line in enumerate(text_str.split('\n')):
        fig.text(0.1, 0.15 - (i * 0.025), line, ha='left')
    #fig.tight_layout()
    fig.show()
    # plt.plot(agenta_lta, label="Agent A")
    # plt.plot(mw_lta, label="MaxWeight")
    # plt.plot(mw_nn_lta, label=f"MW_NN")
    # # get max of all lta
    # max_lta = max([max(agenta_lta), max(mw_lta), max(mw_nn_lta)])
    # plt.ylim(0, 200)
    # plt.title(f"MW NN training comparison for env_id {env_id}")
    # Add an annotation to the plot that gives env_id, arrival rates, mw_nn_error_rate

    # # use ax[2] to add text to the plot
    # ax[2].text(0.5, 0.5, f"env_id = {env_id}")
    # arrival_rates_formatted = [float(f"{x:.2f}") for x in arrival_rates]
    # ax[2].text(0.5, 0.4, f"arrival_rates = {arrival_rates_formatted}")
    # mw_nn_error_rate_formatted = f"{mw_nn_error_rate:.4f}"
    # ax[2].text(0.5, 0.3, f"mw_nn_error_rate = {mw_nn_error_rate_formatted}")
    # w_normalized = [float(f"{x:.4f}") for x in w_normalized]
    # ax[2].text(200, 0.2, f"MW NN Weights = {w_normalized}")
    #
    # plt.legend()
    # plt.show()

    result = {
        "env_id": env_id,
        "arrival_rates": arrival_rates,
        "mw_nn_error_rate": mw_nn_error_rate,
        "w_normalized": w_normalized,
        "mw_lta": mw_lta[-1],
        "mw_nn_lta": mw_nn_lta[-1],
        "agenta_lta": agenta_lta[-1],
    }
    return result


if __name__ == "__main__":
    import pickle
    results = {}
    test_context_set_path = 'SH1_context_set.json'
    #model_path = 'model_2905000.pt'
    mdp_path = "MDP_Solver/saved_mdps/SH1_2_MDP.p"


    maxweight_nn_training_test(env_id=2, context_set_path= test_context_set_path, mdp_path= mdp_path)
    # with open(f"mw_policy_fitting_results_4_1.pkl", 'wb') as f:
    #     pickle.dump(results, f)


    # Get mean of all results for all keys
    # mean_results = {key: np.mean([results[env_id][key] for env_id in results.keys()], axis = 0) for key in results[0].keys()}
    """
    Need to improve the MDP policies, I think the Q_max setting was not large enought
    Maybe there is also an error in the MDP policy
    Performance looked a lot better on previous script so I am thinking there is an error somewhere
    
    """