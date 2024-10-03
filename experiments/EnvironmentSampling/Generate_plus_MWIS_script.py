import networkx as nx
from graph_env_creators import make_line_graph, make_ring_graph, create_grid_graph

from get_graph_characteristics import get_graph_characteristics
import torch
import matplotlib.pyplot as plt
import numpy as np

from modules.torchrl_development.baseline_policies.maxweight import CGSMaxWeightActor
from torchrl.envs.utils import check_env_specs
from modules.torchrl_development.envs.env_creation import make_env_cgs
from modules.torchrl_development.utils.metrics import compute_lta
from modules.torchrl_development.envs.ConflictGraphScheduling import compute_valid_actions

from helpers import create_env_dict
from modules.torchrl_development.utils.configuration import make_serializable
import json


rows = 5
columns = 5
arrival_rate = 0.4
n_rollouts = 3
rollout_length = 20_000

file_name = f"grid_{rows}x{columns}_ar{arrival_rate}.json"

# create a graph
adj, arrival_dist, arrival_rate, service_dist, service_rate = create_grid_graph(rows = rows, columns= columns,  arrival_rate = arrival_rate, service_rate = 1.0)


# create networkx graph from the adjacency matrix
G = nx.convert_matrix.from_numpy_array(adj)
graph_chars = get_graph_characteristics(G)

# Create the environment parameters
env_params = {
    "adj": adj,
    "arrival_dist": arrival_dist,
    "arrival_rate": arrival_rate,
    "service_dist": service_dist,
    "service_rate": service_rate,
    "max_queue_size": 1000,
    "env_type": "CGS"
}

# Create the environment
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
ltas = []
for i in range(n_rollouts):
    # Do 100 steps with actor
    td = env.rollout(max_steps = int(rollout_length), policy = actor)
    ltas.append(compute_lta(td["q"].sum(axis = 1)))
mean_lta = torch.mean(torch.stack(ltas), dim = 0)




"""
PLOT THE RESULTING QUEUE LENGTHS AS A FUNCTION OF TIME
"""
# Plot the lta backlog as a function of time
fig, ax = plt.subplots()
ax.plot(mean_lta)
ax.set_xlabel("Time")
ax.set_ylabel("Mean LTA Backlog")
plt.show()

# Create the environment dictionary
env_dict = create_env_dict(adj, arrival_dist, arrival_rate, service_dist, service_rate, graph_chars, mwis_lta=mean_lta[-1])


serial_env_dict = make_serializable(env_dict)


with open(file_name, "w") as f:
    json.dump(serial_env_dict, f)

# # load json from file and make the environment from it
# with open(file_name, "r") as f:
#     loaded_env_dict = json.load(f)
#
# env = make_env_cgs(loaded_env_dict,
#                 observation_keys=["q", "s"],
#                 seed = 0,
#                   )
#
# td = env.rollout(max_steps = int(500), policy = actor)