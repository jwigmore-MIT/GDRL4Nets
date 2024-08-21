from sklearn.cluster import KMeans
import numpy as np

# Assuming 'arrival_rates' is your list of vectors
arrival_rates = [
    [0.1, 0.1, 0.3, 0.8],
    [0.1, 0.3, 0.3, 0.7],
    [0.2, 0.2, 0.8, 0.2],
    [0.1, 0.3, 0.4, 0.8],
    [0.1, 0.5, 0.6, 0.6],
    [0.2, 0.3, 0.5, 0.8],
    [0.2, 0.8, 0.3, 0.3],
    [0.2, 0.4, 0.7, 0.5],
    [0.3, 0.6, 0.4, 0.3],
    [0.4, 0.3, 0.3, 0.6]
]

# Convert the list of arrival rates into a numpy array for processing
X = np.array(arrival_rates)

# Choose the number of clusters
K = 3  # Example value, adjust based on your needs

# Perform K-means clustering
kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

# The cluster assignments for each arrival rate vector
labels = kmeans.labels_

# Optionally, print out the cluster assignments
for i, cluster in enumerate(labels):
    print(f"Arrival rate {i}: Cluster {cluster}")

# To see the cluster centers (mean arrival rates for each cluster)
print("Cluster centers:")
print(kmeans.cluster_centers_)