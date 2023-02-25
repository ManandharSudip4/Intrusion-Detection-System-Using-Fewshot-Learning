import torch
import csv
import sys
import re
import datetime
import pandas as pd
from frameFunctions import prediction, igtd, extract_sample
from cols_map import cicids_cols, cicflowmeter_cols, colsForMapping

result_path = "AI/Results"
absolute_result_path = "AI/Results/euclidean/data"
n_support = 5
test_x = torch.load("AI/Torches/Testing/30pCom_test_x_euclidean_one55.pt")
test_y = torch.load("AI/Torches/Testing/30pCom_test_y_euclidean_one55.pt")
n_way = len(test_y.unique())

filename = sys.argv[1]
line_no = int(sys.argv[2])
actual_class = "unknown"

if len(sys.argv) == 3:
    data = pd.read_csv(filename, header=None)
    data.columns = cicids_cols
    data = data.astype(float)
    data = data.iloc[[line_no - 1]]
    output_file = "Logs/log_one.txt"
    actual_class = ""

else:
    fileFormat = sys.argv[3]
    with open("uploads/"+filename, "r") as f:
        contents = f.read().strip()
        if re.search("\d", contents.split("\n")[0]):
            df = pd.read_csv("uploads/"+filename, header=None)
            if fileFormat == "cicflowmeter":
                df.columns = cicflowmeter_cols
            else:
                df.columns = cicids_cols
        else:
            df = pd.read_csv("uploads/"+filename)
        if fileFormat == "cicflowmeter":
            df1 = df.rename(columns=colsForMapping)
            data = df1[cicids_cols].astype(float)
        else:
            data = df[cicids_cols].astype(float)
        if line_no:
            data = data.iloc[[line_no - 1]]
            output_file = "Logs/log_one.txt"
        else:
            output_file = "Logs/log_many.txt"

samples = igtd(data, result_path)

count = 0

for index, row in enumerate(samples):
    support = extract_sample(n_way, n_support, 0, test_x, test_y)
    query_path = absolute_result_path + "/_" + row + "_data.txt"
    results, value_of_n = prediction(query_path, support)
    print("-------------------------------------------------")
    print("Result: ", results, "Value of y_hat: ", value_of_n)
    print("-------------------------------------------------")

    if actual_class == "unknown":
        if fileFormat == "cicflowmeter":
            actual_class = ""
        else:
            if "Label" in df.columns:
                if line_no:
                    actual_class = df.iloc[line_no - 1]["Label"]
                else:
                    actual_class = df.iloc[index]["Label"]
            else:
                actual_class = "N/A"

    if not count:
        count += 1
        with open(output_file, "w") as of:
            csv_writer = csv.writer(of)
            if actual_class != "":
                csv_writer.writerow(
                    [
                        str(datetime.datetime.now()).split(".")[0],
                        value_of_n,
                        results,
                        actual_class,
                    ]
                )
            else:
                csv_writer.writerow(
                    [
                        str(datetime.datetime.now()).split(".")[0],
                        value_of_n,
                        results,
                    ]
                )

    else:
        with open(output_file, "a") as of:
            csv_writer = csv.writer(of)
            if actual_class != "":
                csv_writer.writerow(
                    [
                        str(datetime.datetime.now()).split(".")[0],
                        value_of_n,
                        results,
                        actual_class,
                    ]
                )
            else:
                csv_writer.writerow(
                    [
                        str(datetime.datetime.now()).split(".")[0],
                        value_of_n,
                        results,
                    ]
                )
