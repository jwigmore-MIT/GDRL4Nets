import networkx as nx
import random


def generate_strongly_connected_digraph(n, p):
    # Step 1: Create an initial cycle to ensure strong connectivity
    G = nx.DiGraph()
    nodes = list(range(n))

    # Add nodes to the graph
    G.add_nodes_from(nodes)

    # Create a cycle
    for i in range(n):
        G.add_edge(nodes[i], nodes[(i + 1) % n])

    # Step 2: Add additional random edges
    for i in range(n):
        for j in range(n):
            if i != j and not G.has_edge(nodes[i], nodes[j]):
                if random.random() < p:
                    G.add_edge(nodes[i], nodes[j])

    # Ensure the graph is strongly connected
    if not nx.is_strongly_connected(G):
        raise Exception("Generated graph is not strongly connected")

    return G


# Example usage:
n = 10  # Number of nodes
p = 0.3  # Probability of adding an edge

G = generate_strongly_connected_digraph(n, p)

# Draw the graph to visualize (optional)
import matplotlib.pyplot as plt

pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)
plt.show()