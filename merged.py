##### DBSCAN clustering algorithm
#Based on the paper introduced by Martin Ester, Hans-Peter Kriegel, Jörg Sander, Xiaowei Xu. 
#The paper: "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise"

#The following libraries will be imported. Sklearn is not allowed for the exam
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#The Iris public available data set from UCI Machine Learning Repository is uploaded 
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
data = pd.read_csv(url, header=None)

#Preprocess all the data in the first four columns dividing by comma that will be used for clustering 
#We need only digits without names (Iris-setosa, etc.) and without the hedear 
SetOfPoints = data.iloc[:, 0:4].values

#Calculate Euclidean distance between two points to determine whether points are within the radius epsilon or not
def calculate_distance(x, y):
    return np.sqrt(np.sum((x - y)**2))

#Eps-neighborhood of a point. Searching neighbors of a points within a given radius defining by eps
def find_neighbors(data, point_index, eps):

    #while there are unmarked points we use the function to check if there are more points around the current point
    neighbors = []
    for i, point in enumerate(data):
        if i != point_index and calculate_distance(data[point_index], point) < eps:
            neighbors.append(i)
    return neighbors

#Apply DBSCAN algorithm
#At first we say that all the points are noises. Later we will classified them as either core points, border points or noise
def dbscan(data, eps, min_pts):
    labels = np.full(len(data), -1)  
    cluster_label = 0                #setting the counter of all clusters at 0 (Labeling Cluster IDs)

    #Move from one points to another, checking if they are not yet part of any cluster and labeling them as -1 or 0  
    #If these new points have enough neighbors according to min_pts, the new cluster will be created and the points will included in the new clusters 
    
    def expand_cluster(point_index, neighbors):
        for neighbor in neighbors:
            if labels[neighbor] == -1 or labels[neighbor] == 0:
                labels[neighbor] = cluster_label
                new_neighbors = find_neighbors(data, neighbor, eps)
                if len(new_neighbors) >= min_pts:
                    neighbors.extend(new_neighbors)

    for i, point in enumerate(data):
        if labels[i] != -1: #going to the next point if the point is already visited
            continue

        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_pts:
            labels[i] = -1  #mark as noise
        else:
            cluster_label += 1 #increased cluster counter by 1
            labels[i] = cluster_label
            expand_cluster(i, neighbors)
            
    plot_clusters(data, labels, eps, min_pts) #starting the visualisation 


#Visualizing the result in 2D space, using matplotlip library.
def plot_clusters(data, labels, eps, min_pts):
    unique_labels = np.unique(labels)

    for label in unique_labels:
        if label == -1:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], c='black', marker='o', s=30, label='Noise')
        else:
            plt.scatter(data[labels == label, 0], data[labels == label, 1], marker='o', s=30, label=f'Cluster {label}')

    plt.title(f"Iris clusters where eps = {eps:.2f}")
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.legend()
    plt.show()
    print(f"Clusterization with minPts= {min_pts} ran successfully!")


#This function allows manually determine the parameters Eps (epsilon) and MinPts 
#Enter Eps (maximum distance between two points) and minPts (minimum number of points to define a cluster). 
#Our clusterization will be performed for each Eps value in the range depending on the step, start and end value
#Note: a negative or zero value for a step will call the infinite loop and so please do not select it. Only numbers are allowerd.
#Exemple input format: -2,5,0.5,3
def select_eps():
    print("Select start for eps:")
    start_eps = float(input())  
    print("Select end for eps:")
    end_eps = float(input())    
    print("Select the step for eps:")
    step_eps = float(input())   
    current_eps = start_eps
    print("Enter the minimum number of neighboring points required around a reference point(minPts) :")
    min_pts = int(input())
    while current_eps <= end_eps:
        dbscan(SetOfPoints, current_eps, min_pts)
        current_eps += step_eps
        
select_eps()

#Key results and performance evaluation: 
    #The public available data have been clustered based on DBSCAN algorithm introduced in the paper.
    #We can manually set a range of the parameters Eps and minPts. 
    #By running the algorithm with different values of Eps, we can see how these parameters influence the clustering process.
    #Our visualization shows the clusters and noise points in a 2D space
