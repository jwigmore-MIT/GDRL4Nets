import random
import string
from functions import create_context_set_from_folder, subsample_topology_delete_nodes, test_backpressure, delete_nodes
import json
import os
folder_path = "../envs/grid_3x3"
base_file = os.path.join(folder_path, "grid_3x3.json")
env_info = json.load(open(base_file, 'r'))
to_delete = [0]
env_info["class_info"].append({"source": 1, "destination": 8, "rate": 0.4})
new_network_config = delete_nodes(env_info, to_delete)

# run backpressure test
print("Testing backpressure on new network")
mean_lta, tds = test_backpressure(new_network_config, rollout_length=10000, runs=3)
new_network_config["lta"] = mean_lta.item()
# save the new network configuration
with open(base_file.replace(".json", f"_r{'_'.join([str(x) for x in to_delete])}.json"), 'w') as file:
    json.dump(new_network_config, file)


# folder_path = "../envs/grid_4x4_mod"
# context_set_name = "grid_4x4_3_nodes_removed_context_set"
# create_context_set_from_folder(folder_path, context_set_name)

# file_path = "../envs/grid_4x4_mod/grid_4x4.json"
# env_info = json.load(open(file_path, 'r'))
# num_links = len(env_info["link_info"])
# # generation new network configuration
# new_network_config = subsample_topology_delete_nodes(env_info, n_delete=3)
# print("Testing backpressure on new network")
# mean_lta, tds = test_backpressure(new_network_config, rollout_length = 10000, runs = 3)
# env_info["lta"] = mean_lta
# if mean_lta < 250:
#     # generate a random 6 character string with letters and digits
#     appendum = "".join(random.choices(string.ascii_letters + string.digits, k=6))
#     # check if file exists
#     while os.path.exists(file_path.replace(".json", f"_node_subsampled_{appendum}.json")):
#         appendum = "".join(random.choices(string.ascii_letters + string.digits, k=6))
#     # save the new network configuration
#     with open(file_path.replace(".json", f"_node_subsampled_{appendum}.json"), 'w') as file:
#         json.dump(new_network_config, file)


# print("Generating new network configuration")
# new_network_config = subsample_topology_delete_edges(env_info, n_delete=20)
# print("Testing backpressure on new network")
# mean_lta, tds = test_backpressure(new_network_config, rollout_length = 10000, runs = 3)
