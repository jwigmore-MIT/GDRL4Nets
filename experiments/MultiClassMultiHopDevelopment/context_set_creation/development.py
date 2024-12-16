
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
import json
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensordict import TensorDict
import networkx as nx
import matplotlib.pyplot as plt





file_path = "../envs/grid_4x4.json"
# Example usage
with open(file_path, 'r') as file:
    env_info = json.load(file)


net = MultiClassMultiHop(**env_info)

# plot net.graphx with node and edge labels
G = net.graphx
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, font_weight='bold')
plt.show()
td = net.reset()

def subsample_network_from_nodes(env_info, n_delete=1, run_backpressure=True):
    # Sample a new environment info parameters with N_p nodes
    n_delete = 2
    Np = len(env_info["nodes"]) - n_delete
    source_nodes = []
    destination_nodes = []
    for cls in env_info["class_info"]:
        source_nodes.append(cls["source"])
        destination_nodes.append(cls["destination"])

    # create a set of source and destination nodes
    new_nodes = set(source_nodes + destination_nodes)

    # create a set of remaining nodes
    remaining_nodes = set(env_info["nodes"]) - new_nodes

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

    # create new class_info (just a copy)
    new_class_info = []
    for cls in env_info["class_info"]:
        if (node_mapping[cls["source"]] is not None and
                node_mapping[cls["destination"]] is not None):
            new_class_info.append(
                {"source": node_mapping[cls["source"]],
                 "destination": node_mapping[cls["destination"]],
                 "rate": cls["rate"]}
            )

    # create new network configuration dictionary
    new_network_config = {
        "nodes": torch.arange(Np).tolist(),
        "link_distribution": env_info["link_distribution"],
        "arrival_distribution": env_info["arrival_distribution"],
        "link_info": new_link_info,
        "class_info": new_class_info,
        "original_labels": inv_node_mapping
    }

    # create new network environment
    new_net = MultiClassMultiHop(**new_network_config)

    # plot the new network
    G = new_net.graphx
    # label G with original node ids
    nx.relabel_nodes(G, inv_node_mapping, copy=False)

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, font_weight='bold')
    plt.show()

    # Run backpressure on the new network
    if run_backpressure:
        from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
        from modules.torchrl_development.utils.metrics import compute_lta
        bp_actor = BackpressureActor(net=new_net)
        # generate 3 different rollouts
        fig, ax = plt.subplots()
        tds =[]
        ltas = []
        for i in range(3):
            td = new_net.rollout(max_steps=10_000, policy=bp_actor)
            ltas.append(compute_lta(td["Q"].sum((1,2))))
            tds.append(td)
            ax.plot(ltas[-1], label=f"Rollout {i}")
        # plot the mean ltas of the new network
        ltas = torch.stack(ltas)
        mean_lta = ltas.mean(0)
        ax.plot(mean_lta, label="Mean LTA", color="black")

        fig.show()

        new_network_config["lta"] = mean_lta[-1]

    return new_network_config

new_network_config = subsample_network_from_nodes(env_info, n_delete=2, run_backpressure=True)








