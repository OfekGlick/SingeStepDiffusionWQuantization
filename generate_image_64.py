import torch
import pickle
from tqdm import tqdm
import torchvision
import os
import numpy as np

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

traindir = '/home/ofek.glick/BRECQ/imagenet_64/train'
img_size = 64

train_data = []
for i, batch in enumerate(os.listdir(traindir)):
    if not os.path.exists(f"images_dataset"):
        os.mkdir("images_dataset")
    if 'train' not in batch:
        continue
    batch_data = unpickle(os.path.join(traindir, batch))
    x = batch_data['data']
    x = x / np.float32(255)
    x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
    x = torch.tensor(x)
    for j, image in tqdm(enumerate(x)):
        image = image.unsqueeze(0)
        torchvision.utils.save_image(
            image,
            f"images_dataset/image_batch_{batch.split('_')[-1]}_image_{j}.png")

    train_data.append(x)
