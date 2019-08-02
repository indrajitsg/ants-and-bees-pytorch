
import json
import argparse
import torch.nn as nn
import torch.optim as optim
from model.vgg import vgg16
from model.resnet import resnet18
from model.inception import inception_v3

from utils.trainer import train_model
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch
import torch.nn.functional as F
import time
from datetime import datetime
import os
import copy

import pickle
from data_loader.data_loader import get_data_loader
from utils.util import adjust_learning_rate
from utils.util import model_snapshot
from utils.util import ensure_dir


parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument('--resume',help='Resume from previous work')


def main():
    args = parser.parse_args()

    # load runtime configurations
    config = json.load(open("config.json"))
    print("Training with following configurations")
    print("{:<20} {:<10}".format('Key', 'Value'))
    for k, v in config.items():
        print("{:<20} {:<10}".format(k, v))

    # load model
    model = resnet18(pretrained=False, num_classes=2)
    # print(model)

    # define loss criterion
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"],
                          momentum=config["momentum"], weight_decay=config["weight_decay"])

    # define scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config["rop_factor"],
                                  patience=config["rop_patience"], verbose=config["rop_verbose"],
                                  threshold=config["rop_threshold"], threshold_mode='rel',
                                  cooldown=config["rop_cooldown"])


    return None

if __name__ == "__main__":
    main()
