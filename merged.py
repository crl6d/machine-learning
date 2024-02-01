import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(data, point_index, eps):
    return [i for i, point in enumerate(data) if i != point_index and calculate_distance(data[point_index], point) < eps]

def dbscan(data, eps, min_samples):
    labels = np.full(len(data), -1)  # -1 represents noise
    cluster_label = 0

    def expand_cluster(point_index, neighbors):
        for neighbor in neighbors:
            if labels[neighbor] == -1 or labels[neighbor] == 0:
                labels[neighbor] = cluster_label
                new_neighbors = get_neighbors(data, neighbor, eps)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)

    for i, point in enumerate(data):
        if labels[i] != -1:
            continue  # Skip points that have already been visited or are noise

        neighbors = get_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  # Mark as noise
        else:
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(i, neighbors)

    plot_clusters(data, labels)

def plot_clusters(data, labels):
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], c='black', marker='o', s=30, label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], marker='o', s=30, label=f'Cluster {label}')

    plt.title("DBSCAN Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()






# Load the aircraft incident dataset
file_path = "DBSCAN/data-set/crashes.csv"
data = pd.read_csv(file_path)

# Choose features for clustering (you need to decide which columns to use)
features = data[['Aboard', 'Fatalities']].values
# Choose features for 


eps = 30
min_samples = 4

result_labels = dbscan(features, eps, min_samples)
plot_clusters(features, result_labels)


