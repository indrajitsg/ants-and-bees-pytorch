
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model.vgg import vgg16
from model.resnet import resnet18
from model.inception import inception_v3
from model.alexnet import alexnet
from utils.trainer import train_model
from utils.logger import setup_logger
from data_loader.data_loader import get_data_loader
from torch.optim.lr_scheduler import ReduceLROnPlateau

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='Test your model')
parser.add_argument('--resume', type=str2bool, default=False, nargs='?', help='Resume training from checkpoint')
parser.add_argument('--reset_patience', type=str2bool, default=False, nargs='?', help='Reset patience counter to 0')
parser.add_argument('--use_cuda', type=str2bool, default=False, nargs='?', help='Use CUDA for training')
parser.add_argument('--model_type', type=str, help='Model types supported: resnet, vgg, inception')

def main():
    """
    Main function for performing training
    :return: None
    """
    # Parse args
    args = parser.parse_args()
    # load runtime configurations
    config = json.load(open("config.json"))
    # Setup logger
    log = setup_logger(name='Training Log', save_dir='outputs/')
    log.info("Training with following configurations")
    log.info("{:<20} {:<10}".format('Key', 'Value'))
    for k, v in config.items():
        log.info("{:<20} {:<10}".format(k, v))

    print(args)

    # Set device
    if args.use_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            print("Sorry CUDA is not available")
            return None
        else:
            print("Will use CUDA")
    else:
        device = torch.device("cpu")

    # load dataset
    print("Loading data...")
    data_loaders = get_data_loader()
    patience_counter = 0

    # load model
    print("Loading model...")
    if args.model_type == 'vgg':
        model = vgg16(pretrained=False, num_classes=config["num_classes"])
    elif args.model_type == 'inception':
        model = inception_v3(pretrained=False, num_classes=config["num_classes"])
    elif args.model_type == 'alexnet':
        model = alexnet(pretrained=False, num_classes=config["num_classes"])
    else:
        model = resnet18(pretrained=False, num_classes=config["num_classes"])

    model = model.to(device)

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

    if args.resume:
        checkpoint = torch.load(config["checkpoint"])
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        epoch = checkpoint["epoch"]
        best_accuracy = checkpoint["best_accuracy"]
        best_loss = checkpoint["best_loss"]
        train_losses = checkpoint["train_losses"]
        train_accuracy = checkpoint["train_accuracy"]
        val_losses = checkpoint["val_losses"]
        val_accuracy = checkpoint["val_accuracy"]
        elapsed_time = checkpoint["elapsed_time"]
        if not args.reset_patience:
            patience_counter = checkpoint["patience_counter"]
    else:
        epoch = 0
        best_accuracy = 0
        best_loss = 0
        train_losses = []
        train_accuracy = []
        val_losses = []
        val_accuracy = []
        elapsed_time = 0
        patience_counter = 0

    train_model(model = model, data_loaders=data_loaders, device=device, criterion=criterion,
                optimizer=optimizer, scheduler=scheduler, num_epochs=config["num_epochs"],
                patience=config["patience"], epoch=epoch, best_accuracy=best_accuracy,
                best_loss=best_loss, train_losses=train_losses, train_accuracy=train_accuracy,
                val_losses=val_losses, val_accuracy=val_accuracy, elapsed_time=elapsed_time,
                patience_counter=patience_counter, checkpoint=config["checkpoint"],
                best_model=config["best_model"])


    return None

if __name__ == "__main__":
    main()
