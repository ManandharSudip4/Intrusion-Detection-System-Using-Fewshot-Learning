import torch
import pandas as pd
import numpy as np
import time
import os
from IGTD1dto2d.IGTD_Functions import min_max_transform, table_to_image
from model.ProtoNet import ProtoNet

# required for IGTD
num_row = 9
num_col = 9
num = num_row * num_col
save_image_size = 3
max_step = 1000
val_step = 300


# support label
support_label = [
    "Infiltration",
    "DoS Slowhttptest",
    "SSH-Patator",
    "Web Attack � Brute Force",
    "FTP-Patator",
    "DoS Hulk",
    "Bot",
    "Web Attack � XSS",
    "DoS slowloris",
    "DoS GoldenEye",
    "DDoS",
    "BENIGN",
    "PortScan",
    "College_Normal",
    "College_DDOS",
]

model = ProtoNet(
    x_dim=(1, 9, 9),
    hid_dim=16,
    z_dim=256,
).to("cuda:0")

model.load_state_dict(torch.load("AI/Models/model_99"))

model.eval()


def extract_sample(n_way, n_support, n_query, datax, datay):
    sample = 1
    K = datay.unique()
    datay = datay.unsqueeze(1).repeat(1, 9, 1).unsqueeze(2).repeat(1, 1, 9, 1)
    datax = torch.cat((datax, datay), axis=3)
    for cls in K:
        datax_cls = datax[datax[:, 0, 0, 1] == cls.item()]
        datax_cls = datax_cls[:, :, :, 0].unsqueeze(3)
        indexes = torch.randperm(datax_cls.shape[0])
        datax_cls = datax_cls[indexes]
        sample_cls = datax_cls[: (n_support + n_query)]
        sample_cls = sample_cls.float()
        sample_cls = sample_cls.unsqueeze(0)
        if type(sample) == type(1):
            sample = sample_cls
        else:
            sample = torch.vstack((sample, sample_cls))
    sample = sample.permute(0, 1, 4, 2, 3)
    return {
        "sample": sample,
        "n_way": n_way,
        "n_support": n_support,
        "n_query": n_query,
    }


def prediction(query_path, support):
    query = pd.read_csv(query_path, sep="\t", header=None, skipinitialspace=True)
    query = np.array(query)
    query = torch.from_numpy(query)
    query = query.to("cuda:0")
    query = query.float()
    query = torch.unsqueeze(query, 0)
    query = torch.unsqueeze(query, 0)
    query = torch.unsqueeze(query, 0)
    y_hat = model.predict(support["sample"], query, 15, 5, 1)
    value_of_n = y_hat.item()
    results = support_label[value_of_n]
    return results, value_of_n


def igtd(csv_reader, result_path):
    # pre-process
    csv_reader["01"] = 0
    csv_reader["02"] = 0
    csv_reader["03"] = 0
    csv_reader["04"] = 0

    csv_reader.replace([np.inf, -np.inf], np.nan, inplace=True)
    csv_reader.dropna(inplace=True)
    csv_reader = csv_reader.iloc[:, :num]

    norm_data = min_max_transform(csv_reader.values)
    norm_data = pd.DataFrame(
        norm_data, columns=csv_reader.columns, index=csv_reader.index
    )
    # pre-process

    print("Converting [Euclidean] ...")

    t3 = time.time()
    fea_dist_method = "Euclidean"
    image_dist_method = "Euclidean"
    error = "squared"

    result_dir = os.path.join(result_path, "euclidean")
    os.makedirs(name=result_dir, exist_ok=True)
    label = ["Benign", "Data"]  # dummy label
    samples = table_to_image(
        label,
        norm_data,
        [num_row, num_col],
        fea_dist_method,
        image_dist_method,
        save_image_size,
        max_step,
        val_step,
        result_dir,
        error,
    )
    samples = samples.tolist()
    t4 = time.time()

    print(
        f"Time required to convert 1d to 2d [Euclidean]: {t4-t3:.2f}s | {(t4-t3)/60:.2f}m"
    )

    return samples
