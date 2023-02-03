import torch
import pandas as pd
import numpy as np
import os
import random   
def read_images(image_dir, min_images):
    datax = []
    datay = []
    categories = os.listdir(image_dir)
    print(f"Categories:\n{categories}")
    # a = 1
    no_of_cat = 0
    for i, cat in enumerate(categories):
        images = os.listdir(image_dir + "/" + cat)
        image_count = len(images)
        if image_count < min_images:
            continue
        print(f"{no_of_cat} | {cat}")
        for image in images:
            img = pd.read_csv(
                image_dir + "/" + cat + "/" + image,
                sep="\t",
                header=None,
                skipinitialspace=True,
            )
            datax.extend(np.array([img]))
            datay.append(i)
        no_of_cat += 1
        # if a == 3:
        #     break
        # a += 1
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

def extract_query(image_dir):
    categories = os.listdir(image_dir)
    print("Categories: ", categories)
    cat_ind = random.randint(0, len(categories) - 1)
    cat = categories[cat_ind]
    # cat = categories[7]
    print("Cat: ", cat)
    images = os.listdir(image_dir + '/' + cat)
    img_ind = random.randint(0, len(images) - 1)
    img = pd.read_csv(image_dir + '/' + cat + '/' + images[img_ind], sep='\t', header=None, skipinitialspace=True)
    img = np.array(img)
    img = torch.from_numpy(img)
    img = img.to("cuda:0")
    img = img.float()
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0)
    img = torch.unsqueeze(img, 0)
    return img, cat
