import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from alg import *

# Загрузка данных из CSV-файла
file_path = "DBSCAN/data-set/california_housing_test.csv"
data = pd.read_csv(file_path)

# Теперь data - это DataFrame, содержащий данные из CSV-файла

eps = 100
min_samples = 4

result_labels = dbscan(data.values, eps, min_samples)  # передаем массив NumPy, а не DataFrame
plot_clusters(data.values, result_labels)  # передаем массив NumPy, а не DataFrame