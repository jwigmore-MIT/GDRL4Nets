from torchrl_development.envs.env_generators import parse_env_json, make_env
from torchrl_development.actors import MaxWeightActor
import matplotlib.pyplot as plt
from torchrl_development.utils.metrics import compute_lta
from MDP_Solver.SingleHopMDP import SingleHopMDP
from torchrl_development.actors import MDP_module, MDP_actor
import os

# Set the path to the root directory of this file
os.chdir(os.path.realpath("C:\\Users\\Jerrod\\PycharmProjects\\GDRL4Nets\\experiments\\experiment14"))

"""
Copy the code from SH1B_Context_Enumeration.py to test policy iteration
"""
#%%
# Settings
param_key = "a"

rollout_length = 100
q_max = 50
max_vi_iterations = 200
max_pi_iterations = 10
theta = 0.1
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
        }
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

#%% Now create an MDP from the generated environment and load/compute the tx_matrix
mdp = SingleHopMDP(env, name = mdp_name, q_max = q_max)
# check if tx_matrix exists
try:
    mdp.load_tx_matrix(f"tx_matrices/{mdp_name}_qmax{q_max}_discount0.99_computed_tx_matrix.pkl")
except:
    mdp.compute_tx_matrix(f"tx_matrices")
# #%% Load or compute the Value Iteration
# try:
#     mdp.load_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")
# except:
#     mdp.do_VI(max_iterations = max_vi_iterations, theta = theta)
#     mdp.save_VI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_VI_dict.p")

#%% Load or compute the Policy Iteration
try:
    mdp.load_PI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")
except:
    mdp.do_PI(max_iterations = max_pi_iterations, theta = theta)
    mdp.save_PI(f"saved_mdps/{mdp_name}_qmax{q_max}_discount0.99_PI_dict.p")
# %% Create MDP actor
mdp_actor = MDP_actor(MDP_module(mdp))

#%% Test MDP Actor
env = make_env(base_env_params)
mdp_results = {}
mdp_results["td"] = env.rollout(policy=mdp_actor, max_steps = rollout_length)
mdp_results["lta_backlog"] = compute_lta(mdp_results["td"]["backlog"])
mdp_results["q1_lta_backlog"] = compute_lta(mdp_results["td"]["Q"][:,0])
mdp_results["q2_lta_backlog"] = compute_lta(mdp_results["td"]["Q"][:,1])

#%% Plot MDP Actor results against MaxWeight
fig, ax = plt.subplots(1,1)
ax.plot(mw_results["lta_backlog"], label = "MaxWeight Policy (LTA: {:.2f})".format(mw_results["lta_backlog"][-1]))
ax.plot(mdp_results["lta_backlog"], label = "MDP Policy (LTA: {:.2f})".format(mdp_results["lta_backlog"][-1]))
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.set_title(f"MW vs VI Policy: Environment: SH1B{param_key}")
ax.legend()
fig.show()

#%% Plot Q1 and Q2 LTA Backlogs of mdp_results
fig, ax = plt.subplots(1,1)
ax.plot(mdp_results["q1_lta_backlog"], label = "MDP Policy Q1 (LTA: {:.2f})".format(mdp_results["q1_lta_backlog"][-1]))
ax.plot(mdp_results["q2_lta_backlog"], label = "MDP Policy Q2 (LTA: {:.2f})".format(mdp_results["q2_lta_backlog"][-1]))
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

#%% Plot just the Q1 backlog as a function of time
fig, ax = plt.subplots(1,1)
ax.plot(mdp_results["td"]["Q"], label = "MDP Policy Q1")
ax.set_xlabel("Time")
ax.set_ylabel("Backlog")
ax.legend()
fig.show()

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
lta_means = {}
lta_stdevs = {}













