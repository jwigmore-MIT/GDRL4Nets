import heapq
from collections import deque



def compute_shortest_path(network, source_id, destination_id):
    # Dijkstra's algorithm to find the shortest path
    distances = {node_id: float('inf') for node_id in network.nodes}
    previous_nodes = {node_id: None for node_id in network.nodes}
    distances[source_id] = 0
    priority_queue = [(0, source_id)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for edge_id, edge in network.nodes[current_node].edges.items():
            neighbor = edge.tail_node
            distance = current_distance + 1  # Assuming all edges have equal weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Reconstruct the shortest path
    path = []
    current_node = destination_id
    while previous_nodes[current_node] is not None:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    if path:
        path.insert(0, current_node)
    return path

if __name__ == "__main__":
    from file_handling import create_queueing_network
    file_path = "env1.json"
    network, edge_info, traffic_info = create_queueing_network(file_path)

    for source_node in network.source_nodes:
        for destination_id, _, _ in source_node.arrival_processes:
            source_id = source_node.id
            shortest_path = compute_shortest_path(network, source_id, destination_id)
            print(f"Shortest path from {source_id} to {destination_id}: {shortest_path}")