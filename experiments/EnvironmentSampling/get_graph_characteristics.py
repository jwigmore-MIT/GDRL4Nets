import networkx as nx
import numpy as np

def get_graph_characteristics(G):
    graph_chars = {}

    # Get the diameter of G
    graph_chars['diameter'] = nx.diameter(G)

    # Get the average degree of G
    graph_chars['average_degree'] = np.mean(list(dict(G.degree()).values()))

    # Get max degree of G
    graph_chars['max_degree'] = max(list(dict(G.degree()).values()))

    # Get average shortest path length of G
    graph_chars['average_shortest_path'] = nx.average_shortest_path_length(G)

    # Get the number of nodes in G
    graph_chars['num_nodes'] = G.number_of_nodes()

    # Get the number of edges in G
    graph_chars['num_edges'] = G.number_of_edges()

    # Get the number of connected components in G
    graph_chars['num_connected_components'] = nx.number_connected_components(G)

    # Get the degree distribution of G
    graph_chars['degree_distribution'] = nx.degree_histogram(G)

    # Get the density of G
    graph_chars['density'] = nx.density(G)

    # Get eccentricity of G
    # graph_chars['eccentricity'] = nx.eccentricity(G)

    return graph_chars