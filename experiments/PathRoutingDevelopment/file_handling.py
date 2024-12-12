import json
from QueueingNetwork import QueueingNetwork

# Function to load the network from a json file
def load_network_from_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data["edge_info"], data["traffic_info"]

# function to convert the above format to the format that the QueueingNetwork class will take as input

def create_queueing_network(file_path):
    edge_info, traffic_info = load_network_from_json(file_path)
    return QueueingNetwork(edge_info, traffic_info), edge_info, traffic_info

if __name__ == "__main__":
    file_path = "env1.json"
    env, edge_info, traffic_info = create_queueing_network(file_path)
    print(edge_info)
    print(traffic_info)
