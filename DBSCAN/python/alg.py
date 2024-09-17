import numpy as np

def calculate_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

def find_neighbors(data, point_index, eps):
    neighbors = []
    for i, point in enumerate(data):
        if i != point_index and calculate_distance(data[point_index], point) < eps:
            neighbors.append(i)
    return neighbors

def dbscan(data, eps, min_pts):
    labels = np.full(len(data), -1)  
    cluster_label = 0

    def expand_cluster(point_index, neighbors):
        for neighbor in neighbors:
            if labels[neighbor] == -1 or labels[neighbor] == 0:
                labels[neighbor] = cluster_label
                new_neighbors = find_neighbors(data, neighbor, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)

    for i, point in enumerate(data):
        if labels[i] != -1:
            continue

        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_pts:
            labels[i] = -1
        else:
            cluster_label += 1
            labels[i] = cluster_label
            expand_cluster(i, neighbors)
            
    return data, labels