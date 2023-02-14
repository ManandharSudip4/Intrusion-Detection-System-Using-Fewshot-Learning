import torch
import csv
import os
import io
import re
import time
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from frameFunctions import prediction, igtd, extract_sample
from cols_map import cicids_cols, cicflowmeter_cols, colsForMapping


class MyHandler(FileSystemEventHandler):
    def __init__(self, filename):
        super(MyHandler, self).__init__()
        self.filename = filename
        self.filesize = os.stat(filename).st_size

    def on_modified(self, event):
        if event.src_path == self.filename:
            with open(self.filename, "r") as f:
                f.seek(self.filesize)
                new_contents = f.read().strip()
                if new_contents != "":
                    if re.search(r"\d", new_contents.split("\n")[0]):
                        df = pd.read_csv(io.StringIO(new_contents), header=None)
                        df.columns = cicflowmeter_cols
                    else:
                        df = pd.read_csv(io.StringIO(new_contents))
                    df1 = df.rename(columns=colsForMapping)
                    data = df1[cicids_cols].astype(float)
                    samples = igtd(data, result_path)
                    for index, row in enumerate(samples):
                        support = extract_sample(n_way, n_support, 0, test_x, test_y)
                        query_path = absolute_result_path + "/_" + row + "_data.txt"
                        results, value_of_n = prediction(query_path, support)
                        print("-------------------------------------------------")
                        print("Result: ", results, "Value of y_hat: ", value_of_n)
                        print("-------------------------------------------------")
                        with open("Logs/log.txt", "a") as output_file:
                            csv_writer = csv.writer(output_file)
                            csv_writer.writerow(
                                [df1.iloc[index]["timestamp"], value_of_n, results]
                            )
            self.filesize = os.stat(self.filename).st_size


result_path = "AI/Results"
absolute_result_path = "AI/Results/euclidean/data"
n_support = 5
test_x = torch.load("AI/Torches/Testing/30pCom_test_x_euclidean_one55.pt")
test_y = torch.load("AI/Torches/Testing/30pCom_test_y_euclidean_one55.pt")
n_way = len(test_y.unique())

file_to_watch = "./Logs/capture.csv"
event_handler = MyHandler(file_to_watch)
observer = Observer()
observer.schedule(event_handler, file_to_watch)

observer.start()
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()
observer.join()
