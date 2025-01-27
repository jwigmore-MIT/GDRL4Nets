import json
def create_grid_network(grid_size, link_rate, arrival_rate, output_file):
    nodes = list(range(grid_size ** 2))
    link_info = []
    class_info = []

    # Create links for the grid
    for i in range(grid_size):
        for j in range(grid_size):
            node_id = i * grid_size + j
            if j < grid_size - 1:  # Horizontal link
                link_info.append({"start": node_id, "end": node_id + 1, "rate": link_rate})
                # reverse link
                link_info.append({"start": node_id + 1, "end": node_id, "rate": link_rate})
            if i < grid_size - 1:  # Vertical link
                link_info.append({"start": node_id, "end": node_id + grid_size, "rate": link_rate})
                # reverse link
                link_info.append({"start": node_id + grid_size, "end": node_id, "rate": link_rate})
            if i == 0 and j == 0:
                class_info.append({"source": node_id, "destination": grid_size ** 2 - 1, "rate": arrival_rate})
            elif i == grid_size - 1 and j == grid_size - 1:
                class_info.append({"source": node_id, "destination": 0, "rate": arrival_rate})
            elif i == 0 and j == grid_size - 1:
                class_info.append({"source": node_id, "destination": grid_size * (grid_size - 1), "rate": arrival_rate})
            elif i == grid_size - 1 and j == 0:
                class_info.append({"source": node_id, "destination": grid_size - 1 , "rate": arrival_rate})

    # Create traffic classes, where each source is a corner node and the destination is the opposite corner

    # Create the network configuration dictionary
    network_config = {
        "nodes": nodes,
        "link_distribution": "fixed",
        "arrival_distribution": "poisson",
        "link_info": link_info,
        "class_info": class_info
    }

    # Write the configuration to a YAML file
    with open(output_file, 'w') as file:
        json.dump(network_config, file)
    return network_config

# Example usage

grid_size = 5
arrival_rate = 0.1
link_rate = 1
file_path = f"../envs/grid_{grid_size}x{grid_size}_arrival_rate_{arrival_rate}.json"
network_config = create_grid_network(grid_size=grid_size, link_rate=link_rate, arrival_rate=arrival_rate, output_file=file_path)

from experiments.MultiClassMultiHopDevelopment.agents.backpressure_agents import BackpressureActor
from modules.torchrl_development.envs.MultiClassMultihop import MultiClassMultiHop
from matplotlib import pyplot as plt
from modules.torchrl_development.utils.metrics import compute_lta
# import timing for performance evaluation
import time

env = MultiClassMultiHop(**network_config)
bp_actor = BackpressureActor(net=env)
# time the rollouts
start = time.time()
td = env.rollout(max_steps = 50000, policy = bp_actor)
end = time.time()
print(f"Time taken for rollout: {end - start}")
fig, ax = plt.subplots()
ax.plot(compute_lta(-td["next","reward"]))
ax.set_xlabel("Time")
ax.set_ylabel("Queue Length")
ax.legend()
plt.show()

