import pandas as pd
import numpy as np

dataset_path = "/home/user/manualpartition/teamIDS/Datasets/70percent_train_of_all_combined.csv"
# read csv
data = pd.read_csv(dataset_path, low_memory=False, sep=',', engine='c', na_values=['na', '-', ''],  header=0)
data = data.values
data = data[:,68]

# print min and max
print("Column: 1")
print("Min: ", np.min(data))
print("Max: ", np.max(data))
