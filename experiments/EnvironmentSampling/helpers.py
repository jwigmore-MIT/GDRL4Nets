""""
All helper functions for context set creation
"""

import numpy as np
import torch
import os
import json

from torch_geometric.utils import dense_to_sparse


def create_env_dict(adj, arrival_dist, arrival_rate, service_dist, service_rate, graph_chars, mwis_lta = None, greedy_lta = None):

    # Make sure adj is in a readable format
    n_nodes = graph_chars["num_nodes"]
    if adj.shape == (n_nodes, n_nodes): # adj is dense
        adj = dense_to_sparse(torch.Tensor(adj))[0].T
    elif adj.shape[0] == 2 and adj.shape[1] > 2: # adj is sparse
        adj = adj.T


    env_dict = {
        "env_params": {
            "adj": adj,
            "arrival_dist": arrival_dist,
            "arrival_rate": arrival_rate,
            "service_dist": service_dist,
            "service_rate": service_rate,
            "env_type": "CGS",},
        "graph_chars": graph_chars,
        "baselines" :{
            "mwis_lta": mwis_lta,
            "greedy_lta": greedy_lta
        }
    }
    return env_dict


def create_context_set_from_folder(folder: str):
    """
    Create a context set from a folder of environment json files
    Should be a dictionary with keys: "context_dict", "num_environments",
    where context_dict is a dictionary of environment dictionaries
    """
    context_dict = {}
    for n, file in enumerate(os.listdir(folder)):
        if file.endswith(".json"):
            with open(os.path.join(folder, file)) as f:
                env_dict = json.load(f)
                env_dict["name"] = "".join(file.split(".")[:-1])
                context_dict[n] = env_dict

    return {
        "context_dicts": context_dict,
        "num_envs": len(context_dict)
    }

context_set = create_context_set_from_folder("grid_environments")
