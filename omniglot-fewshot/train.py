import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import torch.optim as optim
from tqdm import tqdm
from model.ProtoNet import ProtoNet
from functions import read_images, extract_sample
import os

# check output not understood properly
def display_sample(sample):
    # need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    # make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))

def map_name(image_dir, min_images):
    categories = os.listdir(image_dir)
    # a = 1
    no_of_cat = 0
    for i, cat in enumerate(categories):
        images = os.listdir(image_dir + "/" + cat)
        image_count = len(images)
        if image_count < min_images:
            continue
        print(f"{no_of_cat} {cat}")
        # if a == 3:
        #     break
        # a += 1
        no_of_cat += 1
    
def train(
    model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size
):
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop
    prev_epoch_loss = float("inf")
    acc_loss_df = pd.DataFrame(columns=["epoch", "loss", "accuracy"])

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0
        for episode in tqdm(range(epoch_size), desc=f"Epoch {epoch:d} train"):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print(
            "Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )
        acc_loss_df.loc[len(acc_loss_df.index)] = [epoch, epoch_loss, epoch_acc]
        if epoch_loss < prev_epoch_loss:
            model.save(
                f"/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Models/70pCom_euclidean_one/model_{epoch}"
            )
            prev_epoch_loss = epoch_loss
        epoch += 1
        scheduler.step()
    acc_loss_df.to_csv(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/AccLoss/acc_loss_70pCom_euclidean_one.csv"
    )


if __name__ == "__main__":
    # map_name("/mnt/Documents/Major_Project/few_shot/IGTD-results-benign-downsampled/euclidean/data", 10)
    # exit(0)
    t0 = time.time()
    torch.cuda.empty_cache()
    n_support = 10
    n_query = 10

    print("Reading images...")
    t1 = time.time()
    #train_x, train_y = read_images(
    #    "/home/user/manualpartition/teamIDS/IDS/ai/IGTD-1dto2d/IGTD-results70Com/euclidean/data",
    #    n_support + n_query,
    #)

    train_x = torch.load(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/70pCom_train_x_euclidean_one.pt"
    )
    train_y = torch.load(
        "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/70pCom_train_y_euclidean_one.pt"
    )

    t2 = time.time()

    #torch.save(
    #    train_x,
    #    "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/70pCom_train_x_euclidean_one.pt",
    #)
    #torch.save(
    #    train_y,
    #    "/home/user/manualpartition/teamIDS/IDS/ai/Outputs/Torches/70pCom_train_y_euclidean_one.pt",
    #)

    print(
        f"Time required to read {train_y.size()} files: {t2-t1:.2f}s | {(t2-t1)/60:.2f}m"
    )

    n_way = len(train_y.unique())

    max_epoch = 25
    epoch_size = 4000
    model = ProtoNet(
        x_dim=(1, 9, 9),
        hid_dim=64,
        z_dim=256,
    ).to("cuda:0")

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Training...")
    t3 = time.time()
    train(
        model,
        optimizer,
        train_x,
        train_y,
        n_way,
        n_support,
        n_query,
        max_epoch,
        epoch_size,
    )
    t4 = time.time()
    print(f"Time required to train: {t4-t3:.2f}s | {(t4-t3)/60:.2f}m")
