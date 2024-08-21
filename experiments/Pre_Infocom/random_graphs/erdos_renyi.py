import networkx as nx
import matplotlib.pyplot as plt



n = 10  # Number of nodes
p = 0.3  # Probability of edge creation

# Undirected graph
G_undirected = nx.erdos_renyi_graph(n, p)

# Plot the undirected graph
pos1 = nx.spring_layout(G_undirected)
nx.draw(G_undirected, pos1, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000)
plt.title("Undirected Graph")
plt.show()



# Directed graph
G_directed = nx.gnp_random_graph(n, p, directed=True)

# Plot the directed graph
pos2 = nx.spring_layout(G_directed)
nx.draw(G_directed, pos2, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, arrowsize=20)
plt.title("Directed Graph")
plt.show()
