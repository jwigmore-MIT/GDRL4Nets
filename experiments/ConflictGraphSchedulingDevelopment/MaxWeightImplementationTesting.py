# import

import numpy as np
import torch
from tensordict import TensorDict
from modules.torchrl_development.envs.ConflictGraphScheduling import ConflictGraphScheduling, compute_valid_actions
from modules.torchrl_development.baseline_policies.maxweight import CGSMaxWeightActor
from torchrl.envs.utils import check_env_specs
from modules.torchrl_development.envs.env_creation import make_env_cgs
from modules.torchrl_development.utils.metrics import compute_lta


import matplotlib.pyplot as plt

import pickle

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)







""" 
ENVIRONMENT PARAMETERS
"""
adj = np.array([[0,1,0,0,], [1,0,0,0], [0,0,0,1], [0,0,1,0]])
arrival_dist = "Poisson"
arrival_rate = np.array([0.4, 0.4, 0.4, 0.4])
service_dist = "Fixed"
service_rate = np.ones(adj.shape[0])*1
max_queue_size = 1000

env_params = {
    "adj": adj,
    "arrival_dist": arrival_dist,
    "arrival_rate": arrival_rate,
    "service_dist": service_dist,
    "service_rate": service_rate,
    "max_queue_size": max_queue_size,
    "env_type": "CGS"
}

"""
CREATE THE ENVIRONMENT
"""
env = make_env_cgs(env_params,
               observation_keys=["q", "s"],
               seed = 0,
              )

check_env_specs(env)


""""
CREATE THE CGSMAXWEIGHT ACTOR
"""
actor = CGSMaxWeightActor(valid_actions = compute_valid_actions(env))


"""
TEST THE MAXWEIGHT ACTOR
"""
# Do 100 steps with actor
td = env.rollout(max_steps = int(10000), policy = actor)
lta = compute_lta(td["q"].sum(axis = 1))



"""
PLOT THE RESULTING QUEUE LENGTHS AS A FUNCTION OF TIME
"""
fig, ax = plt.subplots()
for i in range(env.num_nodes):
    ax.plot(td["q"][:,i], label=f"Queue {i}")
ax.set_xlabel("Time")
ax.set_ylabel("Queue Length")
ax.legend()
plt.show()

# Plot the lta backlog as a function of time
fig, ax = plt.subplots()
ax.plot(lta)
ax.set_xlabel("Time")
ax.set_ylabel("LTA Backlog")
plt.show()


"""
SAVE THE ROLLOUT DATA
"""

# pickle.dump(td, open("maxweight_actor_rollout.pkl", "wb"))
