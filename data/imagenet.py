import os
import torch
import torchvision
from tqdm import tqdm
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset
import pickle
import numpy as np


def build_imagenet64_data(
        data_path: str = '',
        img_size: int = 64,
        input_size: int = 224,
        batch_size: int = 64,
        workers: int = 4,
        dist_sample: bool = False,
):
    print('==> Using Pytorch Dataset')

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    train_data = []
    train_labels = []
    for i, batch in tqdm(enumerate(os.listdir(traindir))):
        if 'train' not in batch:
            continue
        batch_data = unpickle(os.path.join(traindir, batch))
        x = batch_data['data']
        y = batch_data['labels']
        mean_image = batch_data['mean']

        # Normalize images
        x = x / np.float32(255)
        mean_image = mean_image / np.float32(255)
        x -= mean_image
        x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
        y = torch.tensor([i - 1 for i in y])
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        x = torch.tensor(x)

        train_data.append(x)
        train_labels.append(y)

    train_data = torch.cat(train_data, dim=0)
    train_labels = torch.cat(train_labels, dim=0)

    train_dataset = TensorDataset(train_data, train_labels)

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    return train_loader


def build_imagenet_data(
        data_path: str = '',
        input_size: int = 224,
        batch_size: int = 64,
        workers: int = 4,
        dist_sample: bool = False,
):
    print('==> Using Pytorch Dataset')

    traindir = os.path.join(data_path, 'train')
    valdir = os.path.join(data_path, 'val')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    # torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]))

    if dist_sample:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        sampler=val_sampler,
    )
    return train_loader, val_loader


def save_imagenet_64_files_as_png(
        data_path: str = '',
        img_size: int = 64,
):
    print('==> Using Pytorch Dataset')
    from PIL import Image
    traindir = os.path.join(data_path, 'train')

    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo)
        return dict

    train_data = []
    print("Starting to save images")
    for i, batch in tqdm(enumerate(os.listdir(traindir))):
        if 'train' not in batch:
            continue
        batch_data = unpickle(os.path.join(traindir, batch))
        print("preprocessing")
        x = batch_data['data']
        y = batch_data['labels']
        x = x / np.float32(255)
        x = np.dstack((x[:, :4096], x[:, 4096:8192], x[:, 8192:]))
        x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
        x = torch.tensor(x)
        print("done preprocessing")
        for j, image in tqdm(enumerate(x)):
            torchvision.utils.save_image(
                image,
                fp=f"/home/ofekglick/BRECQ/imagenet_64_images/image_batch_{batch.split('_')[-1]}_image_{j}.png",
            )
            print("Finished saving images of batch {}".format(batch.split('_')[-1]))
        train_data.append(x)


if __name__ == '__main__':
    save_imagenet_64_files_as_png(
        data_path=os.path.join('/home/ofekglick/BRECQ', 'imagenet_64'),
    )
