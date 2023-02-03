import numpy as np
import pandas as pd

# from scipy import ndimage
import matplotlib.pyplot as plt

# import multiprocessing as mp
import os

# import cv2
import time
import random

import torch
import torchvision
import torch.optim as optim

# from tqdm.notebook import trange
# from tqdm import tqdm_notebook
from tqdm import tqdm

from model.ProtoNet import ProtoNet

# def read_alphabets(alphabet_directory_path, alphabet_directory_name):
#     """
#     Reads all the characters from a given alphabet_directory
#     """
#     datax = []
#     datay = []
#     characters = os.listdir(alphabet_directory_path)
#     for character in characters:
#         images = os.listdir(alphabet_directory_path + character + '/')
#         for img in images:
#             image = cv2.resize(
#                 cv2.imread(alphabet_directory_path + character + '/' + img),
#                 (28,28)
#                 )
#             #rotations of image
#             rotated_90 = ndimage.rotate(image, 90)
#             rotated_180 = ndimage.rotate(image, 180)
#             rotated_270 = ndimage.rotate(image, 270)
#             datax.extend((image, rotated_90, rotated_180, rotated_270))
#             datay.extend((
#                 alphabet_directory_name + '_' + character + '_0',
#                 alphabet_directory_name + '_' + character + '_90',
#                 alphabet_directory_name + '_' + character + '_180',
#                 alphabet_directory_name + '_' + character + '_270'
#             ))
#     return np.array(datax), np.array(datay)

# def read_images(base_directory, min_imgaes):
#     """
#     Reads all the alphabets from the base_directory
#     Uses multithreading to decrease the reading time drastically
#     """
#     datax = None
#     datay = None
#     pool = mp.Pool(mp.cpu_count())
#     results = [pool.apply(read_alphabets,
#                           args=(
#                               base_directory + '/' + directory + '/', directory,
#                               )) for directory in os.listdir(base_directory)]
#     pool.close()
#     for result in results:
#         if datax is None:
#             datax = result[0]
#             datay = result[1]
#         else:
#             datax = np.vstack([datax, result[0]])
#             datay = np.concatenate([datay, result[1]])
#     return datax, datay

########################################## need changes ################
# def read_images():
#    pass
def read_images(image_dir, min_images):
    datax = []
    datay = []
    categories = os.listdir(image_dir)
    print(f"Categories:\n{categories}")
    a = 1
    for i, cat in enumerate(categories):
        images = os.listdir(image_dir + "/" + cat)
        image_count = len(images)
        if image_count < min_images:
            continue
        # t0 = time.time()
        for image in images:
            img = pd.read_csv(
                image_dir + "/" + cat + "/" + image,
                sep="\t",
                header=None,
                skipinitialspace=True,
            )
            datax.extend(np.array([img]))
            datay.extend(np.array([i]))

        # t1 = time.time()
        # datay = np.array(datay)
        # print(f'Time to read {datay.size} files: {t1-t0}')
        # exit(0)
        if a == 2:
            break
        a += 1
    datax = np.array(datax)
    datax = torch.from_numpy(datax)
    datax = datax.to("cuda:0")
    datax = torch.unsqueeze(datax, -1)

    datay = np.array(datay)
    datay = np.vstack(datay).astype(float)
    datay = torch.from_numpy(datay)
    datay = datay.to("cuda:0")

    print(f"Is Cuda test:\ndatax {datax.is_cuda}\ndatay {datay.is_cuda}")
    return datax, datay


def read_categorical_images_limited(data_dir, data_length):
    datax = []
    datay = []
    categories = os.listdir(data_dir)
    for i, cat in enumerate(categories):
        images = os.listdir(data_dir + "/" + cat)
        image_count = len(images)
        if image_count < data_length:
            continue
        images = random.shuffle(os.listdir())
        for j in range(data_length):
            img = pd.read_csv(
                data_dir + "/" + cat + "/" + images[j],
                sep="\t",
                header=None,
                skipinitialspace=True,
            )
            datax.extend(np.array([img]))
            datay.extend(np.array([i]))
    datax = np.array(datax)
    datax = torch.from_numpy(datax)
    datax = datax.to("cuda:0")
    datax = torch.unsqueeze(datax, -1)

    datay = np.array(datay)
    datay = np.vstack(datay).astype(float)
    datay = torch.from_numpy(datay)
    datay = datay.to("cuda:0")
    return datax, datay


# def extract_sample(n_way, n_support, n_query, datax, datay):
#     sample = []
#     # not needed tho only needed in training??? later
#     K = np.random.choice(np.unique(datay), n_way, replace=False)
#     for cls in K:
#         datax_cls = datax[datay == cls]
#         perm = np.random.permutation(datax_cls)
#         sample_cls = perm[:(n_support+n_query)]
#         sample.append(sample_cls)
#     sample = np.array(sample)
#     sample = torch.from_numpy(sample).float()
#     # change dimension [4, 6, 2, 3, 2] => [4, 6, 2, 2, 3]
#     sample = sample.permute(0, 1, 4, 2, 3)
#     return ({
#         'sample': sample,
#         'n_way': n_way,
#         'n_support': n_support,
#         'n_query': n_query
#     })


def extract_sample_limited(n_way, n_support, n_query, data_dir):
    sample = 1
    datax, datay = read_categorical_images_limited(data_dir, "cat", n_support + n_query)
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


# check output not understood properly
def display_sample(sample):
    # need 4D tensor to create grid, currently 5D
    sample_4D = sample.view(sample.shape[0] * sample.shape[1], *sample.shape[2:])
    # make a grid
    out = torchvision.utils.make_grid(sample_4D, nrow=sample.shape[1])
    plt.figure(figsize=(16, 7))
    plt.imshow(out.permute(1, 2, 0))


################################
def train_limited(
    model, optimizer, data_dir, n_way, n_support, n_query, max_epoch, epoch_size
):
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc=f"Epoch {epoch:d} train"):
            sample = extract_sample_limited(n_way, n_support, n_query, data_dir)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print(
            "Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )
        epoch += 1
        scheduler.step()


def train(
    model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size
):
    # divide the learning rate by 2 at each epoch, as suggested in paper
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
    epoch = 0  # epochs done so far
    stop = False  # status to know when to stop

    while epoch < max_epoch and not stop:
        running_loss = 0.0
        running_acc = 0.0

        for episode in tqdm(range(epoch_size), desc=f"Epoch {epoch:d} train"):
            sample = extract_sample(n_way, n_support, n_query, train_x, train_y)
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(sample)
            running_loss += output["loss"]
            running_acc += output["acc"]
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_acc / epoch_size
        print(
            "Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}".format(
                epoch + 1, epoch_loss, epoch_acc
            )
        )
        epoch += 1
        scheduler.step()


def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        sample = extract_sample(n_way, n_support, n_query, test_x, test_y)
        loss, output = model.set_forward_loss(sample)
        running_loss += output["loss"]
        running_acc += output["acc"]
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("Test results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


def test_limted(model, data_dir, n_way, n_support, n_query, test_episode):
    running_loss = 0.0
    running_acc = 0.0
    for episode in tqdm(range(test_episode)):
        sample = extract_sample(n_way, n_support, n_query, data_dir)
        loss, output = model.set_forward_loss(sample)
        running_loss += output["loss"]
        running_acc += output["acc"]
    avg_loss = running_loss / test_episode
    avg_acc = running_acc / test_episode
    print("Test results -- Loss: {:.4f} Acc: {:.4f}".format(avg_loss, avg_acc))


if __name__ == "__main__":
    t0 = time.time()
    torch.cuda.empty_cache()
    model = ProtoNet(
        x_dim=(1, 9, 9),
        hid_dim=64,
        z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_dir = "/mnt/Documents/Major_Project/few_shot/IGTD-results/euclidean/data"

    n_way = 2
    n_support = 5
    n_query = 5

    max_epoch = 5
    epoch_size = 2000

    # print('Training Limited...')
    # train_limited(model, optimizer, data_dir, n_way, n_support, n_query, max_epoch, epoch_size)

    print("Reading images...")
    t1 = time.time()
    train_x, train_y = read_images(
        "/mnt/Documents/Major_Project/few_shot/IGTD-results/euclidean/data",
        n_support + n_query,
    )
    t2 = time.time()
    print(
        f"Time required to read {train_y.size()} files: {t2-t1:.2f}s | {(t2-t1)/60:.2f}m"
    )

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
