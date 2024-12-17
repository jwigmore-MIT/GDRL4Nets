
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt
from copy import deepcopy
import os
from typing import Union, List
def get_link_map(link_info):
    link_map = {}
    for id,link_dict in enumerate(link_info):
        link_map[id] = (link_dict["start"], link_dict["end"])
    return link_map, list(link_map.values())

from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
from modules.torchrl_development.utils.metrics import compute_lta
def test_backpressure(env_info, rollout_length = 10000, runs = 3):
    # Create environment
    net = MultiClassMultiHop(**env_info)
    # Create backpressure actor
    bp_actor = BackpressureActor(net=net)
    # generate 3 different rollouts
    fig, ax = plt.subplots()
    tds =[]
    ltas = []
    for i in range(runs):
        td = net.rollout(max_steps=rollout_length, policy=bp_actor)
        ltas.append(compute_lta(td["Q"].sum((1,2))))
        tds.append(td)
        ax.plot(ltas[-1], label=f"Rollout {i}")
    # plot the mean ltas of the new network
    ltas = torch.stack(ltas)
    mean_lta = ltas.mean(0)
    ax.plot(mean_lta, label="Mean LTA", color="black")
    fig.show()
    return mean_lta[-1], tds

def check_paths(G, s_d_pairs):
    # check if there is a path between all source destination pairs
    for s, d in s_d_pairs:
        try:
            has_path = nx.shortest_path(G, s, d)
        except:
            return False
        if not has_path:
            return False
    return True


def subsample_topology_delete_edges(env_info, n_delete=1):
    # Sample a new environment info parameters by deleting n_delete edges from the original
    # If a node is now disconnected, remove it as well
    success = False # flag to indicate if the subsampling was successful
    while not success:
        original_edges = env_info["link_info"]
        # sample n_delete random indices
        indices = torch.randperm(len(original_edges))
        delete_edges = set(indices[:n_delete].tolist())
        # create a set of remaining edges
        new_link_info = []
        for m in range(len(original_edges)):
            if m not in delete_edges:
                new_link_info.append(original_edges[m])
        link_map, edge_list = get_link_map(new_link_info)
        # create a new graph object with the new link_info
        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        # plot G

        # Check if there is a path between all source destination pairs
        source_nodes = []
        destination_nodes = []
        for cls in env_info["class_info"]:
            source_nodes.append(cls["source"])
            destination_nodes.append(cls["destination"])
        s_d_pairs = list(zip(source_nodes, destination_nodes))
        success = check_paths(G, s_d_pairs)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()
    # create a new network configuration dictionary
    new_network_config = {
        "nodes": env_info["nodes"],
        "link_distribution": env_info["link_distribution"],
        "arrival_distribution": env_info["arrival_distribution"],
        "link_info": new_link_info,
        "class_info": env_info["class_info"]
    }
    return new_network_config

def delete_nodes(env_info, node_ids: Union[int, List[int]]):
    """
    Deletes the specified node from the original node info, link info, and class info
    :param env_info:
    :param node_id:
    :return:
    """
    if isinstance(node_ids, int):
        node_ids = [node_ids]
    delete_nodes = set(node_ids)
    # create a dictionary that maps the old node indices to the new node indices
    node_mapping = {}
    n_new = 0
    for i, node in enumerate(env_info["nodes"]):
        if node not in delete_nodes:
            node_mapping[node] = n_new
            n_new += 1
        else:
            node_mapping[node] = None
    # Get the inverse mapping
    inv_node_mapping = {v: k for k, v in node_mapping.items() if v is not None}
    # create new link_info
    new_link_info = []
    for m, link_info in enumerate(env_info["link_info"]):
        if (node_mapping[link_info["start"]] is not None and
                node_mapping[link_info["end"]] is not None):
            new_link_info.append({
                "start": node_mapping[link_info["start"]],
                "end": node_mapping[link_info["end"]],
                "rate": link_info["rate"]
            })

    new_class_info = []
    for cls in env_info["class_info"]:
        if (node_mapping[cls["source"]] is not None and
                node_mapping[cls["destination"]] is not None):
            new_class_info.append(
                {"source": node_mapping[cls["source"]],
                 "destination": node_mapping[cls["destination"]],
                 "rate": cls["rate"]}
            )
        else:
            print(f"Class {cls} was removed due to node deletion")

    link_map, edge_list = get_link_map(new_link_info)
    # create a new graph object with the new link_info
    G = nx.DiGraph()
    G.add_edges_from(edge_list)
    # create new network configuration dictionary
    new_network_config = {
        "nodes": torch.arange(len(inv_node_mapping.keys())).tolist(),
        "link_distribution": env_info["link_distribution"],
        "arrival_distribution": env_info["arrival_distribution"],
        "link_info": new_link_info,
        "class_info": new_class_info,
        "original_labels": inv_node_mapping
    }

    # label G with original node ids
    nx.relabel_nodes(G, new_network_config["original_labels"], copy=False)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()

    return new_network_config


def subsample_topology_delete_nodes(env_info, n_delete=1, keep_class_nodes = True):
    # Sample a new environment info parameters with N_p nodes
    Np = len(env_info["nodes"]) - n_delete
    if keep_class_nodes:
        source_nodes = []
        destination_nodes = []
        for cls in env_info["class_info"]:
            source_nodes.append(cls["source"])
            destination_nodes.append(cls["destination"])

        # create a set of source and destination nodes
        new_nodes = set(source_nodes + destination_nodes)
    else:
        new_nodes = set()

    # create a set of remaining nodes
    remaining_nodes = set(env_info["nodes"]) - new_nodes
    success = False  # flag to indicate if the subsampling was successful
    while not success:
        # select n_delete nodes to delete from remaining nodes
        new_nodes = list(new_nodes)
        remaining_nodes = torch.tensor(list(remaining_nodes))
        indices = torch.randperm(remaining_nodes.size(0)).tolist()
        delete_nodes = set(remaining_nodes[indices[:n_delete]].tolist())

        # create a dictionary that maps the old node indices to the new node indices
        node_mapping = {}
        n_new = 0
        for i, node in enumerate(env_info["nodes"]):
            if node not in delete_nodes:
                node_mapping[node] = n_new
                n_new += 1
            else:
                node_mapping[node] = None
        # Get the inverse mapping
        inv_node_mapping = {v: k for k, v in node_mapping.items() if v is not None}

        # create new link_info
        new_link_info = []
        for m, link_info in enumerate(env_info["link_info"]):
            if (node_mapping[link_info["start"]] is not None and
                    node_mapping[link_info["end"]] is not None):
                new_link_info.append({
                    "start": node_mapping[link_info["start"]],
                    "end": node_mapping[link_info["end"]],
                    "rate": link_info["rate"]
                })

        new_class_info = []
        for cls in env_info["class_info"]:
            if (node_mapping[cls["source"]] is not None and
                    node_mapping[cls["destination"]] is not None):
                new_class_info.append(
                    {"source": node_mapping[cls["source"]],
                     "destination": node_mapping[cls["destination"]],
                     "rate": cls["rate"]}
                )

        link_map, edge_list  = get_link_map(new_link_info)
        # create a new graph object with the new link_info
        G = nx.DiGraph()
        G.add_edges_from(edge_list)
        if keep_class_nodes:
        # check if there is a path between all source destination pairs
            source_nodes = []
            destination_nodes = []
            for cls in new_class_info:
                source_nodes.append(cls["source"])
                destination_nodes.append(cls["destination"])
            s_d_pairs = list(zip(source_nodes, destination_nodes))
            success = check_paths(G, s_d_pairs)
        else:
            success = True

    # create new network configuration dictionary
    new_network_config = {
        "nodes": torch.arange(Np).tolist(),
        "link_distribution": env_info["link_distribution"],
        "arrival_distribution": env_info["arrival_distribution"],
        "link_info": new_link_info,
        "class_info": new_class_info,
        "original_labels": inv_node_mapping
    }


    # label G with original node ids
    nx.relabel_nodes(G, new_network_config["original_labels"], copy=False)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()



    return new_network_config


def create_context_set_from_folder(folder_path, context_set_name):
    """
    Creates a context set from all the files contained within a particular folder
    :param folder_path:
    :param context_set_name:
    :return:
    """
    # get all the files in the folder
    files = os.listdir(folder_path)
    save_path = os.path.join(folder_path, context_set_name)
    context_set = {}
    n = 0
    for file in files:
        if file.endswith(".json"):
            env_info = json.load(open(os.path.join(folder_path, file), 'r'))
            context_set[n] = env_info
            n+=1
    # save the context set
    with open(f"{save_path}.json", 'w') as file:
        json.dump(context_set, file)
    return context_set



# if name == "main":
if __name__ == "__main__":
    import random
    import string

    folder_path = "../envs/grid_4x4_mod"
    context_set_name = "grid_4x4_3_nodes_removed_context_set"
    create_context_set_from_folder(folder_path, context_set_name)

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






