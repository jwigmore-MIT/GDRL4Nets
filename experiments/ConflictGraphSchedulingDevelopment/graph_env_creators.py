import numpy as np

def make_line_graph(n, arrival_rate, service_rate):
    """
    Creates a line graph with n nodes
    :param n:
    :param arrival_rate:
    :param service_rates:
    :return:
    """
    adj = np.zeros((n,n))
    for i in range(n-1):
        adj[i, i+1] = 1
        adj[i+1, i] = 1

    arrival_dist = "Bernoulli"
    arrival_rate = np.ones(n) * arrival_rate
    service_dist = "Fixed"
    service_rates = np.ones(n) * service_rate
    return adj, arrival_dist, arrival_rate, service_dist, service_rates

def make_ring_graph(n, arrival_rate, service_rate):
    adj = np.zeros([n,n])

    adj[0,n-1] = 1
    for i in range(n-1):
        adj[i, i + 1] = 1
        adj[i + 1, i] = 1
    adj[n-1,0] = 1
    arrival_dist = "Bernoulli"
    arrival_rate = np.ones(n) * arrival_rate
    service_dist = "Fixed"
    service_rates = np.ones(n) * service_rate
    return adj, arrival_dist, arrival_rate, service_dist, service_rates

def create_grid_graph(rows = 2, columns = 3, arrival_rate = 0.4, service_rate = 1.0):

    adj_dict = {}
    # for i in range(width):
    #     for j in range(height):
    #         if i < width - 1:
    #             adj_dict[(i, j)].append((i + 1, j))
    #         if j < height - 1:
    #             adj_dict[(i, j)].append((i, j + 1))
    def make_matrix(rows, cols):
        n = rows * cols
        M = np.zeros([n, n])
        for r in range(rows):
            for c in range(cols):
                i = r * cols + c
                # Two inner diagonals
                if c > 0: M[i - 1, i] = M[i, i - 1] = 1
                # Two outer diagonals
                if r > 0: M[i - cols, i] = M[i, i - cols] = 1
        return M
    # adj_dict = {
    #        0: [1,3],
    #        1: [0, 2, 4],
    #        2: [1, 5],
    #        3: [0, 4],
    #        4: [1,3,5],
    #        5: [2,4]}
    # n = len(adj_dict.keys())
    # adj = np.zeros([n, n])
    # for key in adj_dict.keys():
    #     for val in adj_dict[key]:
    #         adj[key, val] = 1
    adj = make_matrix(rows, columns)
    n = adj.shape[0]
    arrival_dist = "Bernoulli"
    arrival_rate = np.ones(n) * arrival_rate
    service_dist = "Fixed"
    service_rates = np.ones(n) * service_rate
    return adj, arrival_dist, arrival_rate, service_dist, service_rates

