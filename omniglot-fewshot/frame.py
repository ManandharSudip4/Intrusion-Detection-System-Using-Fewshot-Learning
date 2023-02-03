import os
import sys
sys.path.append("/home/user/manualpartition/teamIDS/IDS/ai")
from IGTD1dto2d.IGTD_Functions import min_max_transform, table_to_image
from frameFunctions import prediction, igtd, extract_sample
import numpy as np
import time
import csv
import random
import pandas as pd
import torch
from model.ProtoNet import ProtoNet
                    
# All required paths
dataset_path = "csv_used/predict.csv"  
result_path = "IGTD-results_for_omniglot/"
absolute_result_path = "IGTD-results_for_omniglot/euclidean/data"

# length
current_len = 0
last_len = 0

# for Prediction
n_support = 5

# loading model
test_x = torch.load(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_x_euclidean_one55.pt",
    )
test_y = torch.load(
    "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/Testing/30pCom_test_y_euclidean_one55.pt",
)
n_way = len(test_y.unique())

while True:
    if current_len == 0:
        try:
            print("Data Reading: UnSkipp")
            csv_reader = pd.read_csv(dataset_path, low_memory=False, sep=',', engine='c', na_values=['na', '-', ''], header=0)
            label = csv_reader['Label']
            csv_reader.drop('Label', axis=1, inplace=True)

            current_len = len(csv_reader)
            print("Current length Updated 1: ", current_len)
        except Exception as e:
            print("Error Message 1: ", e)
    else:
        try:
            print("Data Reading: Skipp")
            csv_reader = pd.read_csv(dataset_path, low_memory=False, sep=',', engine='c', na_values=['na', '-', ''],  header=0, skiprows=range(0,current_len))
            csv_readerr = pd.read_csv(dataset_path, header=None)
            
            current_len = len(csv_readerr)
            print("Current length Updated 2: ", current_len)
        except Exception as e:
            print("Error Message 3: ", e)
            break 
    try:
        samples = igtd(csv_reader, result_path)
        
        for row in samples:
            # .txt files can be created here
            # prediction block
            support = extract_sample(n_way, n_support, 0, test_x, test_y)
            query_path = absolute_result_path + '/_' + row + '_data.txt'
            results, value_of_n = prediction(query_path, support)
            print("-------------------------------------------------")
            print("Result: ", results, "Value of y_hat: ", value_of_n)
            print("-------------------------------------------------")
            # Write the results to a new CSV file
            with open("output.csv", "a") as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow([f"Time {row}", results])
    except Exception as e:
        print("Error Message 2: ", e)
    time.sleep(10) # Sleep for 10 seconds before checking for new rows again



