import random
import string
from functions import create_context_set_from_folder, subsample_topology_delete_nodes, test_backpressure, delete_nodes, plot_network
import json
import os
folder_path = "../envs/grid_4x4_mod"
to_exclude = ["grid_4x4.json", "grid_4x4_3_nodes_removed_context_set.json"]
"""
For each .json file in the folder, except for files listed in exlude_files, 
1. create the environment
2. run backpressure
3. add the lta to the environment configuration
4. save the environment configuration
5. create a new context set from all the files in the folder
"""

for file in os.listdir(folder_path):
    if file.endswith(".json") and file not in to_exclude:
        base_file = os.path.join(folder_path, file)
        new_network_config = json.load(open(base_file, 'r'))
        plot_network(new_network_config, title=file.split(".")[0])
        # run backpressure test
        print("Testing backpressure on new network")
        mean_lta, tds = test_backpressure(new_network_config, rollout_length=10000, runs=3)
        new_network_config["lta"] = mean_lta.item()
        # save the new network configuration
        with open(base_file, 'w') as file:
            json.dump(new_network_config, file)

# create context set
context_set_name = "grid_4x4_3_nodes_removed_context_set"
create_context_set_from_folder(folder_path, context_set_name, exlude_files=["grid_4x4_3_nodes_removed_context_set.json"])