__all__ = ["get_data_loader"]

import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import data_loader.customized_transforms as ct
import os
import json

# define data loader config path
data_loader_config_path = os.path.join(os.path.dirname(__file__), "config.json")

# define transforms
data_transforms = {
    'train': transforms.Compose([
        ct.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        ct.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

def get_data_loader():
    """Create data loader
    """
    # load json file
    config = json.load(open(data_loader_config_path))
    # data_dir = os.path.join(os.path.dirname(__file__), config['data_dir'])
    data_dir = config['data_dir']
    batch_size = config['batch_size']
    num_workers = config['num_workers']
    train_pct = config['train_val_split']
    val_pct = 1-train_pct
    rnd_split_seed = config['seed']


    source_data = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                       transform=data_transforms['train'])

    test_data = datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                     transform=data_transforms['val'])

    # compute size of training data based on pct split
    trn_size = int(train_pct * len(source_data))

    # split train into train and val
    torch.manual_seed(rnd_split_seed)
    indices = torch.randperm(len(source_data))
    train_indices = indices[:trn_size or None]
    valid_indices = indices[trn_size:]

    # create data loaders
    train_loader = DataLoader(source_data, pin_memory=True,
                              batch_size=batch_size,
                              sampler=SubsetRandomSampler(train_indices),
                              num_workers=num_workers)

    val_loader = DataLoader(source_data, pin_memory=True,
                            batch_size=batch_size,
                            sampler=SubsetRandomSampler(valid_indices),
                            num_workers=num_workers)

    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)

    data_loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    return data_loaders


def imshow(input, title=None):
    """
    Visualize the pics after data augmentation
    :param input:
        input: pytorch image batch tensor with shape (batch_size, color_channel, W, H)
    :param title:
    :return:
        None
    """
    input = torchvision.utils.make_grid(input)
    input = input.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    input = std * input + mean
    input = np.clip(input, 0, 1)
    plt.imshow(input)
    if title is not None:
        plt.title(title)
    plt.savefig("pics_after_data_augmentation.jpg")
