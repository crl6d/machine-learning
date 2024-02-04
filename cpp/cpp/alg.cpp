#include <iostream>
#include <vector>
#include <cmath>

class DBSCAN {
public:
    DBSCAN(const std::vector<std::vector<double>>& data, double eps, int min_pts);
    void run();
    void printClusters();

private:
    std::vector<std::vector<double>> dataset;
    std::vector<int> labels;
    double epsilon;
    int minPoints;
    int clusterLabel;

    double calculateDistance(const std::vector<double>& x, const std::vector<double>& y);
    std::vector<int> findNeighbors(int pointIndex);
    void expandCluster(int pointIndex, const std::vector<int>& neighbors);
};

DBSCAN::DBSCAN(const std::vector<std::vector<double>>& data, double eps, int min_pts)
    : dataset(data), epsilon(eps), minPoints(min_pts), clusterLabel(0)
{
    labels.resize(dataset.size(), -1);
}

double DBSCAN::calculateDistance(const std::vector<double>& x, const std::vector<double>& y) {
    double distance = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        distance += std::pow(x[i] - y[i], 2);
    }
    return std::sqrt(distance);
}

std::vector<int> DBSCAN::findNeighbors(int pointIndex) {
    std::vector<int> neighbors;
    for (size_t i = 0; i < dataset.size(); ++i) {
        if (i != static_cast<size_t>(pointIndex) && calculateDistance(dataset[pointIndex], dataset[i]) < epsilon) {
            neighbors.push_back(static_cast<int>(i));
        }
    }
    return neighbors;
}

void DBSCAN::expandCluster(int pointIndex, const std::vector<int>& neighbors) {
    for (int neighbor : neighbors) {
        if (labels[neighbor] == -1 || labels[neighbor] == 0) {
            labels[neighbor] = clusterLabel;
            std::vector<int> newNeighbors = findNeighbors(neighbor);
            if (!newNeighbors.empty()) {
                expandCluster(neighbor, newNeighbors);
            }
        }
    }
}

void DBSCAN::run() {
    for (size_t i = 0; i < dataset.size(); ++i) {
        if (labels[i] != -1) {
            continue;
        }

        std::vector<int> neighbors = findNeighbors(static_cast<int>(i));

        if (neighbors.size() < minPoints) {
            labels[i] = -1; // mark as noise
        }
        else {
            ++clusterLabel;
            labels[i] = clusterLabel;
            expandCluster(static_cast<int>(i), neighbors);
        }
    }
}

void DBSCAN::printClusters() {
    for (int label : labels) {
        std::cout << label << " ";
    }
    std::cout << std::endl;
}

// main.cpp

#include <iostream>
#include <vector>
#include "alg.cpp" // Include the implementation file

int main() {
    std::vector<std::vector<double>> data = {
        {5.1, 3.5, 1.4, 0.2},
        {4.9, 3.0, 1.4, 0.2},
        {4.7, 3.2, 1.3, 0.2},
        // ... add more data points
    };

    double eps;
    int minPts;

    std::cout << "Enter the value for epsilon (eps): ";
    std::cin >> eps;

    std::cout << "Enter the value for minPts: ";
    std::cin >> minPts;

    DBSCAN dbscan(data, eps, minPts);
    dbscan.run();

    std::cout << "Cluster labels: ";
    dbscan.printClusters();

    return 0;
}