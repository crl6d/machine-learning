import matplotlib.pyplot as plt
import pandas as pd
import alg
import numpy as np


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None)
SetOfPoints = data.iloc[:, 0:4].values

def plot_clusters(eps_values, data, labels, min_pts):
    for i, eps in enumerate(eps_values):
        plt.figure(figsize=(7, 5))  # Set your desired size for each window
        unique_labels = np.unique(labels[i])

        for label in unique_labels:
            if label == -1:
                plt.scatter(data[i][labels[i] == label, 0], data[i][labels[i] == label, 1], c='black', marker='o', s=30, label='Noise')
            else:
                plt.scatter(data[i][labels[i] == label, 0], data[i][labels[i] == label, 1], marker='o', s=30, label=f'Cluster {label}')

        plt.title(f"Iris clusters where eps = {eps:.2f}")
        plt.xlabel("x-label")
        plt.ylabel("y-label")
        plt.legend()
        plt.show()

    print(f"Clusterization with minPts={min_pts} ran successfully!")

def select_eps():
    print("Select start for eps:")
    start_eps = float(input())  
    print("Select end for eps:")
    end_eps = float(input())    
    print("Select the step for eps:")
    step_eps = float(input())   
    eps_values = np.arange(start_eps, end_eps + step_eps, step_eps)
    print("Enter the minimum number of neighboring points required around a reference point(minPts):")
    min_pts = int(input())

    data_list = []
    labels_list = []
    for eps in eps_values:
        current_data, current_labels = alg.dbscan(SetOfPoints, eps, min_pts)
        data_list.append(current_data)
        labels_list.append(current_labels)

    plot_clusters(eps_values, data_list, labels_list, min_pts)

if __name__ == "__main__":
    select_eps()