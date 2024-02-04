

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def find_neighbors(data, point_index, eps):
    #while there are unmarked points we use the function to check if there are more points around the current point
    neighbors = []
    for i, point in enumerate(data):
        if i != point_index and calculate_distance(data[point_index], point) < eps:
            neighbors.append(i)
    return neighbors




def dbscan(data, eps, min_samples):
    labels = np.full(len(data), -1)  # at first we say that all the points are noises.(later we will change it)
    cluster_label = 0                # setting the counter of all clusters at 0

    def expand_cluster(point_index, neighbors):
        for neighbor in neighbors:
            if labels[neighbor] == -1 or labels[neighbor] == 0:
                labels[neighbor] = cluster_label
                new_neighbors = find_neighbors(data, neighbor, eps)
                if len(new_neighbors) >= min_samples:
                    neighbors.extend(new_neighbors)

    for i, point in enumerate(data):
        if labels[i] != -1: #going to the next point if the point is already visited
            continue

        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            labels[i] = -1  #mark as noise
        else:
            cluster_label += 1 #increased cluster counter by 1
            labels[i] = cluster_label
            expand_cluster(i, neighbors)

    plot_clusters(data, labels) #starting the visualisation 




#visualizing the result, using matplotlip library
def plot_clusters(data, labels):
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], c='black', marker='o', s=30, label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], marker='o', s=30, label=f'Cluster {label}')

    plt.title("DBSCAN Clustering")
    plt.xlabel("x-lable")
    plt.ylabel("y-lable")
    plt.legend()
    plt.show()


# file_path = "../Sample_Data/crashes.csv"
# data = pd.read_csv(file_path)

# # Choose features for clustering
# features = data[['Aboard', 'Fatalities']].values

# eps = 10
# min_samples = 4


# result_labels = dbscan(features, eps, min_samples)
# plot_clusters(features, result_labels)
# print("the code ran successfuly!")

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
iris_data = pd.read_csv(url, header=None)

# Preprocess all the data but without the fourth colomn because we need only digints to be processed
iris_features = iris_data.iloc[:, 0:4].values

# Define DBSCAN parameters
eps = 0.5  # Epsilon value for Iris dataset
min_samples = 2  # Minimum samples for Iris dataset

# Apply DBSCAN clustering
iris_labels = dbscan(iris_features, eps, min_samples)

# Plot the results


