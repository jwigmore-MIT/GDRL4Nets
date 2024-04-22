from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from torchrl_development.actors import MDP_module, MDP_actor
import torch


"""
The goal of this file is to create a network instance with a small effective state-action space under the MaxWeight and optimal policy.
It is a manual script that will import the topology from the SH1B.json environment file, and allow me to change the 
arrival rates and service rates of the network.  The SH1B network is a 2 queue network with Bernoulli arrivals of rate
$\lambda_i$ and Bernoulli capacities with rate $\mu_i$. 
"""
#%%
# Settings
param_key = "b"

rollout_length = 10000
q_max = 50
max_vi_iterations = 200
vi_theta = 0.1
eval_rollouts = 3
eval_seeds = [1,2,3]

# Configure Base Params
base_env_params = parse_env_json("SH1B.json")
mdp_name = f"SH1B{param_key}"
#
param_dict = {
    "a":
        {
            "X_params":
                {
                    "1": {"arrival": [0, 1], "probability": [0.5, 0.5]},
                    "2": {"arrival": [0, 1], "probability": [0.5, 0.5]}
                },
            "Y_params":
                {
                    "1": {"capacity": [0, 2], "probability": [0.7, 0.3]},
                    "2": {"capacity": [0, 3], "probability": [0.7, 0.3]}
                }
        },

    "b": {
        "X_params":
            {
                "1": {"arrival": [0, 1], "probability": [0.5, 0.5]},
                "2": {"arrival": [0, 1], "probability": [0.5, 0.5]}
            },
        "Y_params":
            {
                "1": {"capacity": [0, 1], "probability": [0.3, 0.7]},
                "2": {"capacity": [0, 3], "probability": [0.3, 0.7]}
            }
    },
}


## Override the arrival rates
base_env_params["X_params"]['1']['arrival'] = param_dict[param_key]["X_params"]['1']['arrival']
base_env_params["X_params"]['1']['probability'] = param_dict[param_key]["X_params"]['1']['probability']
base_env_params["X_params"]['2']['arrival'] = param_dict[param_key]["X_params"]['2']['arrival']
base_env_params["X_params"]['2']['probability'] = param_dict[param_key]["X_params"]['2']['probability']

## Override the service rates
base_env_params["Y_params"]['1']['capacity'] = param_dict[param_key]["Y_params"]['1']['capacity']
base_env_params["Y_params"]['1']['probability'] = param_dict[param_key]["Y_params"]['1']['probability']
base_env_params["Y_params"]['2']['capacity'] = param_dict[param_key]["Y_params"]['2']['capacity']
base_env_params["Y_params"]['2']['probability'] = param_dict[param_key]["Y_params"]['2']['probability']


# Make Environment
env = make_env(base_env_params)

# Create MaxWeight Actor
mw_actor = MaxWeightActor(in_keys=["Q", "Y"], out_keys=["action"])
#%%
# Generate Trajectory under MaxWeight Policy
mw_results = {}
mw_results["td"] = env.rollout(policy=mw_actor, max_steps = rollout_length)
mw_results["lta_backlog"] = compute_lta(mw_results["td"]["backlog"])
mw_results["q1_lta_backlog"] = compute_lta(mw_results["td"]["Q"][:,0])
mw_results["q2_lta_backlog"] = compute_lta(mw_results["td"]["Q"][:,1])

#%%
# Plot the LTA Backlog
fig, ax = plt.subplots(1,1)
ax.plot(mw_results["lta_backlog"], label = "MaxWeight Policy (LTA: {:.2f})".format(mw_results["lta_backlog"][-1]))
ax.plot(mw_results["q1_lta_backlog"], label = "Q1 LTA Backlog (LTA: {:.2f})".format(mw_results["q1_lta_backlog"][-1]))
ax.plot(mw_results["q2_lta_backlog"], label = "Q2 LTA Backlog (LTA: {:.2f})".format(mw_results["q2_lta_backlog"][-1]))
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.set_title(f"Environment: SH1B{param_key}")
ax.legend()
fig.show()

#%% Now create an MDP from the generated environment
mdp = SingleHopMDP(env, name = mdp_name, q_max = q_max)
# check if tx_matrix exists
try:
    mdp.load_tx_matrix(f"tx_matrices/{mdp_name}_qmax{q_max}_discount0.99_computed_tx_matrix.pkl")
except:
    mdp.compute_tx_matrix(f"tx_matrices")
try:
    mdp.load_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
except:
    mdp.do_VI(max_iterations = max_vi_iterations, theta = vi_theta)
    mdp.save_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
# %% Create MDP actor
mdp_actor = MDP_actor(MDP_module(mdp))


# %% evaluate both the MaxWeight and MDP policies over three different rollouts/seeds
results = {}
for policy_name, actor in {"MaxWeight":mw_actor, "MDP": mdp_actor}.items():
    policy_results = {}
    for seed in eval_seeds:
        env = make_env(base_env_params, seed = seed)
        td = env.rollout(policy=actor, max_steps = rollout_length)
        lta = compute_lta(td["backlog"])
        print(f"Actor: {policy_name}, Seed: {seed}, LTA: {lta[-1]}")
        policy_results[seed] = {"td": td, "lta": lta}
    results[policy_name] = policy_results

# %% Plot the results
for policy_name, policy_results in results.items():
    all_ltas = torch.stack([torch.tensor(policy_results[seed]["lta"]) for seed in eval_seeds])
    mean_lta = all_ltas.mean(dim = 0)
    std_lta = all_ltas.std(dim = 0)
    results[policy_name]["mean_lta"] = mean_lta
    results[policy_name]["std_lta"] = std_lta

fig, ax = plt.subplots(1,1)
for policy_name, policy_results in results.items():
    mean_lta = policy_results["mean_lta"]
    std_lta = policy_results["std_lta"]
    ax.plot(policy_results["mean_lta"], label = f"{policy_name} Policy")
    ax.fill_between(range(len(mean_lta)), mean_lta - std_lta, mean_lta + std_lta, alpha = 0.1)
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()














