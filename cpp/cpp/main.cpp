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

    double;
    int;

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