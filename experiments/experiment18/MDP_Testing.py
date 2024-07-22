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

def plot_state_action_heatmap(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], lim = 30):
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
    x_axis = x_axis[1:lim+1]
    y_axis = y_axis[1:lim+1]

    action_values = np.zeros((len(x_axis), len(y_axis)))
    for i, x in enumerate(x_axis):
        for j, y in enumerate(y_axis):
            action_values[i, j] = new_df[(new_df[axis_keys[0]] == x) & (new_df[axis_keys[1]] == y)]["Action"]
    ax.imshow(action_values, cmap = 'plasma')
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


def plot_state_action_map(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = None, lim = 30):
    # increase fontsize for the x and y axis labels and ticks
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
    sc = ax.scatter(new_df[axis_names[0]], new_df[axis_names[1]], c = color, cmap = 'bwr', s= 100) # how to increase marker size? s = 100
    # sc = ax.imshow
    # fig.colorbar(sc, ax = ax, label = "Action 2 Probability", ticks = [0,1,2])
    if "Y1" in axis_names:
        ax.set_ylim(-0.5, 2.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_xticks(range(0, 3))
        ax.set_yticks(range(0, 3))
    elif "lam1" in axis_names:
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
    else:
        ax.set_ylim(-0.5, lim+1)
        ax.set_xlim(-0.5, lim+1)
        ax.set_xticks(range(0, lim + 1, 5))
        ax.set_yticks(range(0, lim + 1, 5))
    axis_label_dict = {"Q1": f"$q_1$",
                         "Q2": f"$q_2$",
                         "Y1": f"$y_1$",
                         "Y2": f"$y_2$",
                         "lam1": f"$\lambda_1$",
                         "lam2": f"$\lambda_2$",
                         "mu1": f"$\mu_1$",
                         "mu2": f"$\mu_2$"}
    ax.set_xlabel(axis_label_dict[axis_names[0]])
    ax.set_ylabel(axis_label_dict[axis_names[1]])
    # set x axis label frequency

    hold_names = ", ".join([f"{x[0]}={x[1]}" for x in hold_tuples])
    # combine hold names to create a string
    policy_add = "Deterministic" if pt == "Action" else "Stochastic"
    title = f"{policy_type} {policy_add} Policy for {hold_names}"
    if collected_frames is not None:
        title = title + f" (Collected Frames: {collected_frames})"
    title = hold_names
    ax.set_title(title)



def plot_state_action_map_comparison(df, hold_tuples, ax = None, axis_keys = ["Q1", "Q2"], policy_type = "MLP", plot_type = "Action_Probs", collected_frames = None, lim = 30):
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

def plot_single_mdp_policy(mdp, lim = 20, y_val = (1,1)):
    # use latex for the title labels
    #plt.rc('text', usetex=True)
    policy_table = mdp.pi_policy.policy_table
    df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])
    df["Action"] = list(policy_table.values())
    df["Action"] = df["Action"].apply(lambda x: np.argmax(x))
    df = df[(df["Y1"] == y_val[0]) & (df["Y2"] == y_val[1])]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_state_action_map(df, [("Y1", y_val[0]), ("Y2", y_val[1])], ax=ax, axis_keys=["Q1", "Q2"], policy_type="PI",
                           plot_type="Action", lim=lim)
    # plot_state_action_heatmap(df, [("Y1", 1), ("Y2", 1)], ax=ax, axis_keys=["Q1", "Q2"], lim=lim)
    # Set the title to be \mathbf c^{(mdp.name)} with the mdp.name as a superscript
    ax.set_title(f"$\mathbf{{{'c'}}}^{{({mdp.name})}}$ \n $ \mathbf{{{'y'}}} = {y_val}$ , $\lambda_1 = {{{mdp.env.base_env.arrival_rates[0]}}}$, $\mu_1 = {{{mdp.env.base_env.service_rates[0]}}}$")

    plt.show()
    return fig

def plot_single_mdp_y_policy(mdp, lim = 2, q_val = (5,5)):
    policy_table = mdp.pi_policy.policy_table
    df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])
    df["Action"] = list(policy_table.values())
    df["Action"] = df["Action"].apply(lambda x: np.argmax(x))
    df = df[(df["Q1"] == q_val[0]) & (df["Q2"] == q_val[1])]
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plot_state_action_map(df, [("Q1", q_val[0]), ("Q2", q_val[1])], ax=ax, axis_keys=["Y1", "Y2"], policy_type="PI",
                            plot_type="Action", lim=lim)
    ax.set_title(f"$\mathbf{{{'c'}}}^{{({mdp.name})}}$ \n $ \mathbf{{{'q'}}} = {q_val}$ , $\lambda_1 = {{{mdp.env.base_env.arrival_rates[0]}}}$, $\mu_1 = {{{mdp.env.base_env.service_rates[0]}}}$")
    plt.show()
    return fig


def plot_multi_mdp_policy(mdps: dict):
    # first we want to create a policy table with columns Q1, Q2, Y1, Y2, lam1, lam2, mu1, mu2, action
    df = pd.DataFrame()
    for mdps_key, mdp in mdps.items():
        policy_table = mdp.pi_policy.policy_table
        temp_df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])
        temp_df["lam1"], temp_df["lam2"] = mdp.env.base_env.arrival_rates
        temp_df["mu1"], temp_df["mu2"] = mdp.env.base_env.service_rates
        temp_df["Action"] = list(policy_table.values())
        temp_df["Action"] = temp_df["Action"].apply(lambda x: np.argmax(x))
        temp_df["MDP"] = mdps_key
        # concatentate to the main df
        df = pd.concat([df, temp_df], axis = 0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    hold_tuples= [("Y1", 1), ("Y2", 1), ("Q1", 5), ("Q2", 7)]
    # for tup in hold_tuples:
    #     df = df[df[tup[0]] == tup[1]]
    # want to plot over the dimension of lam1 and mu1 for fixed lam2 and mu2
    plot_state_action_map(df, hold_tuples, ax=ax, axis_keys=["lam1", "mu1"], policy_type="PI", plot_type="Action")
    plt.show()
    return fig





if __name__ == "__main__":
    plt.rcParams.update({'font.size': 20})
    # use latex for the title labels
    # plt.rc('text', usetex=True)

    mdps = {}
    name_dict = {"base": 1,
                 "a": 2,
                 "b": 3,
                 "c": 4,
                 "d": 5,
                 "e": 6}


    for mdp in ["base", "a", "b", "c", "d", 'e']:
        mdps[mdp] = get_mdp(mdp, 40)
        mdps[mdp].name = name_dict[mdp]

    # plot_multi_mdp_policy({"base": mdps["base"],"a": mdps["a"], "b": mdps["b"], "c": mdps["c"], "d": mdps["d"]})

    for mdp in mdps.values():
        fig = plot_single_mdp_policy(mdp)
        # save the figure as a PDF
        fig.savefig(open(f"saved_figs/SingleHopMDP_{mdp.name}_Policy.pdf", "wb"), format = "pdf")
    fig = plot_single_mdp_y_policy(mdps["e"], q_val = (5,5))
    fig.savefig(open(f"saved_figs/SingleHopMDP_{mdps['e'].name}_Y_Policy.pdf", "wb"), format = "pdf")
    fig = plot_single_mdp_policy(mdps["e"], y_val = (1,1))
    fig.savefig(open(f"saved_figs/SingleHopMDP_{mdps['e'].name}_Y1=1_Y2=1_Policy.pdf", "wb"), format = "pdf")
    fig = plot_single_mdp_policy(mdps["e"], y_val = (1,2))
    fig.savefig(open(f"saved_figs/SingleHopMDP_{mdps['e'].name}_Y1=1_Y2=2_Policy.pdf", "wb"), format = "pdf")
    fig = plot_single_mdp_policy(mdps["e"], y_val = (2,1))
    fig.savefig(open(f"saved_figs/SingleHopMDP_{mdps['e'].name}_Y1=2_Y2=1_Policy.pdf", "wb"), format = "pdf")
    fig = plot_single_mdp_policy(mdps["e"], y_val = (2,2))
    fig.savefig(open(f"saved_figs/SingleHopMDP_{mdps['e'].name}_Y1=2_Y2=2_Policy.pdf", "wb"), format = "pdf")



    #
    # # Plot the state-action map for the PI policy
    # policy_table = mdps['a'].pi_policy.policy_table
    # # policy_table = mdp.get_VI_policy()
    # df = pd.DataFrame(policy_table.keys(), columns=["Q1", "Q2", "Y1", "Y2"])
    # df["Action"] = list(policy_table.values())
    # df["Action"] = df["Action"].apply(lambda x: np.argmax(x))
    # df = df[(df["Y1"] == 1) & (df["Y2"] == 1)]
    # fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    # plot_state_action_map(df, [("Y1", 1), ("Y2", 1)], ax=ax, axis_keys=["Q1", "Q2"], policy_type="PI",
    #                       plot_type="Action", lim=20)
    # plt.show()
    #
    #
    # # Get the MaxWeight policy
    # mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
    #
    # # Get mdp_actor
    # mdp_actor = MDP_actor(MDP_module(mdp, policy_type = "PI"))
    #
    # # Evalutate both the PI and MW policies
    # # results = {}
    # # results = {}
    # # for policy_name, actor in {"PI": mdp_actor, "MW": mw_actor}.items():
    # #     policy_results = {}
    # #     for seed in eval_seeds:
    # #         env = make_env(base_env_params, seed=seed)
    # #         td = env.rollout(policy=actor, max_steps=rollout_length)
    # #         lta = compute_lta(td["backlog"])
    # #         print(f"Actor: {policy_name}, Seed: {seed}, LTA: {lta[-1]}")
    # #         policy_results[seed] = {"td": td, "lta": lta}
    # #     results[policy_name] = policy_results
    # #
    # # for policy_name, policy_results in results.items():
    # #     all_ltas = torch.stack([torch.tensor(policy_results[seed]["lta"]) for seed in eval_seeds])
    # #     mean_lta = all_ltas.mean(dim=0)
    # #     std_lta = all_ltas.std(dim=0)
    # #     results[policy_name]["mean_lta"] = mean_lta
    # #     results[policy_name]["std_lta"] = std_lta
    # #
    # # fig, ax = plt.subplots(1, 1)
    # # for policy_name, policy_results in results.items():
    # #     mean_lta = policy_results["mean_lta"]
    # #     std_lta = policy_results["std_lta"]
    # #     ax.plot(policy_results["mean_lta"], label=f"{policy_name} Policy")
    # #     ax.fill_between(range(len(mean_lta)), mean_lta - std_lta, mean_lta + std_lta, alpha=0.1)
    # ax.set_xlabel("Time")
    # ax.set_ylabel("Backlog")
    # ax.legend()
    # fig.show()




