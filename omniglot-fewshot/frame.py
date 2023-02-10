import torch
import pandas as pd
import csv
import datetime
from frameFunctions import prediction, igtd, extract_sample
import os
import sys
# sys.path.append("/home/user/manualpartition/teamIDS/IDS/ai")

# All required paths
# dataset_path = "csv_used/predict.csv"
result_path = "AI/Results"
absolute_result_path = "AI/Results/euclidean/data"

# length
# current_len = 0
# last_len = 0

# for Prediction
n_support = 5

# loading model
test_x = torch.load(
    "AI/Torches/Testing/30pCom_test_x_euclidean_one55.pt",
)
test_y = torch.load(
    "AI/Torches/Testing/30pCom_test_y_euclidean_one55.pt",
)
n_way = len(test_y.unique())

data = sys.argv[1].split(",")
print("In python...")
data = [data[i:i + 77] for i in range(0, len(data), 77)]
csv_reader = pd.DataFrame(data).astype(float)
print(csv_reader)
print(os.getcwd())
samples = igtd(csv_reader, result_path)

count = 0

for row in samples:
    # .txt files can be created here
    # prediction block
    support = extract_sample(n_way, n_support, 0, test_x, test_y)
    query_path = absolute_result_path + "/_" + row + "_data.txt"
    results, value_of_n = prediction(query_path, support)
    print("-------------------------------------------------")
    print("Result: ", results, "Value of y_hat: ", value_of_n)
    print("-------------------------------------------------")

    # Write the results to a new CSV file
    if (len(sys.argv) == 3):
        if (sys.argv[2] == 1):
            with open("Logs/log_one.txt", "w") as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerow(
                    [int(datetime.datetime.now().timestamp() * 1000), value_of_n, results])
        else:
            if not count:
                count += 1
                with open("Logs/log_many.txt", "w") as output_file:
                    csv_writer = csv.writer(output_file)
                    csv_writer.writerow(
                        [int(datetime.datetime.now().timestamp() * 1000), value_of_n, results])
            else:
                with open("Logs/log_many.txt", "a") as output_file:
                    csv_writer = csv.writer(output_file)
                    csv_writer.writerow(
                        [int(datetime.datetime.now().timestamp() * 1000), value_of_n, results])
    else:
        with open("Logs/log.txt", "a") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerow(
                [int(datetime.datetime.now().timestamp() * 1000), value_of_n, results])
