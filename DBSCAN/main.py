import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alg import *

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