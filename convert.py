import numpy as np
import h5py
import os

import multiprocessing
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from scipy import io

from tqdm import tqdm


image_size = 224
resize = transforms.Resize((image_size, image_size))
num_cpus = multiprocessing.cpu_count() - 1
imagenet_dataset = datasets.ImageFolder("./train") # to keep labels aligned with ImageFolder way


def process(f):
    with Image.open(f) as img:
        im = resize(img.convert("RGB"))
    im = np.array(im)
    im = np.transpose(im, (2, 0, 1))
    return im


mat = io.loadmat("meta.mat")
wnid = [mat['synsets']['WNID'][i][0][0] for i in range(1000)]

categories = list(map(lambda x : os.path.join('train', x), wnid))
train_labels = []
for i, c in enumerate(categories):
    files = list(map(lambda x : os.path.join(c, x), os.listdir(c)))
    train_labels += [imagenet_dataset.class_to_idx[wnid[i]] for _ in range(len(files))]
train_shape = (len(train_labels), 3, image_size, image_size)
train_labels = np.array(train_labels, dtype='int16')


val_files = list(map(lambda x : os.path.join('val', x), os.listdir('val')))
val_files.sort()
with open("ILSVRC2012_validation_ground_truth.txt", "r") as file:
    val_labels = np.array([imagenet_dataset.class_to_idx[wnid[int(l.strip()) - 1]] for l in file.readlines()], dtype='int16')
val_shape = (len(val_files), 3, image_size, image_size)


with h5py.File('imagenet-224.hdf5', 'w') as f:

    train_group = f.create_group("train")
    val_group = f.create_group("val")
    train_group.create_dataset("label", data=train_labels)
    val_group.create_dataset("label", data=val_labels)
    del val_labels

    train_group.create_dataset("data", shape=train_shape, dtype=np.uint8)
    val_group.create_dataset("data", shape=val_shape, dtype=np.uint8)
    
    pool = multiprocessing.Pool(num_cpus)
    i = 0
    for j, c in tqdm(enumerate(categories), total=len(categories), ncols=80):
        train_data = np.array(pool.map(process, list(map(lambda x : os.path.join(c, x), os.listdir(c)))), dtype='uint8')
        train_group['data'][i:i+train_data.shape[0]] = train_data
        i += train_data.shape[0]
    del train_labels

    f['val']['data'][:] = np.array(pool.map(process, val_files), dtype='uint8')

