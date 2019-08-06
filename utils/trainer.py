import time
import torch
import matplotlib.pyplot as plt

__all__ = ['train_model']

def _fit(stage, model, data_loader, batch, device, criterion, optimizer):
    """
    Model fit / eval function
    """
    if stage == 'train':
        model.train(True)
    elif stage == 'eval':
        model.train(False)
        model.eval()
    else:
        print('Error')

    total_loss = 0
    total_acc = 0

    # Iterate through batches
    for i, data in enumerate(data_loader):
        if i % 10 == 0:
            print("\r{} batch {}/{}".format(stage, i, batch), end='',
                  flush=True)

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
#         print(outputs.shape)
#         print(labels.shape)
        labels = labels.view(-1, )
        loss = criterion(outputs, labels)
#         print("\n")
#         print(outputs.shape)
#         print(labels.shape)
#         print(type(loss))
        if stage == 'train':
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total_acc += preds.eq(labels).sum().double()

        del(inputs, labels, outputs, preds)
        torch.cuda.empty_cache()

    return total_loss, total_acc, model


def train_model(model, data_loaders, device, criterion, optimizer, scheduler,
                num_epochs, patience, epoch, best_accuracy, best_loss,
                train_losses, train_accuracy, val_losses, val_accuracy,
                elapsed_time, patience_counter, checkpoint, best_model):

    """
    Model trainer function. Incorporates scheduler, resume, early stopping,
    temp_scaling.

    Args:
        model         : the model object
        train_loader  : the train data loader. No train-validation splitting
                        will happen internally
        val_loader    : the validation data loader. Used to check accuracy and
                        perform early stopping
        device        : the type of device. Should be 'cuda' if GPU is
                        available
        criterion     : loss criterion to use. Eg. nn.CrossEntropyLoss()
        optimizer     : optimizer to use. Eg. torch.optim.SGD()
        scheduler     : scheduler for reducing the learning rate at predefined
                        epochs. Eg. torch.optim.lr_scheduler.StepLR()
        resume        : True/False if you are resuming training
        new_model     : Path to new model to be saved
        num_epochs    : number of epochs (int) default 50
        patience      : number of epochs to wait before early stopping kicks
                        in, default 10

    Returns:
        None          : The function saves the model state dict in the
                        specified folder and prints accuracy and loss plots
                        (training and validation)

    """

    # start timer
    since = time.time()

    # define patience counter for early stopping
    patience_counter = patience_counter

    # Initialize best accuracy to 0
    best_acc = best_accuracy
    best_loss = best_loss

    # Compute batch size - used only in printing log
    train_batches = len(data_loaders['train'])
    val_batches = len(data_loaders['val'])

    # Capture size of the data sets to compute average loss and accuracy
    train_size = 0
    val_size = 0

    for d, l in data_loaders['train']:
        train_size = train_size + len(d)

    for d, l in data_loaders['val']:
        val_size = val_size + len(d)

    # Starting epochs
    for epoch in range(epoch, num_epochs):

        # Check for auto stop
        with open('utils/autostop.txt', 'rt') as f:
            content = f.readlines()
            if int(content[0]) == 1:
                break

        print("Epoch {}/{}".format(epoch, num_epochs))
        print('-' * 10)

        # Print current learning rate
        print("Learning Rate: {}".format(optimizer.param_groups[0]['lr']))

        # -------------- Model training -------------- #

        loss_train, acc_train, model = _fit('train', model, data_loaders['train'],
                                            train_batches, device, criterion, optimizer)

        print()
        avg_loss = loss_train / train_size
        avg_acc = acc_train / train_size
        train_losses.append(avg_loss)
        train_accuracy.append(avg_acc)

        # -------------- Model validation --------------#

        loss_val, acc_val, model = _fit('eval', model, data_loaders['val'], val_batches,
                                        device, criterion, optimizer)

        avg_loss_val = loss_val / val_size
        avg_acc_val = acc_val / val_size
        val_losses.append(avg_loss_val)
        val_accuracy.append(avg_acc_val)

        print()
        print("Epoch {} result: ".format(epoch))
        print("Avg loss (train): {:.4f}".format(avg_loss))
        print("Avg acc (train): {:.4f}".format(avg_acc))
        print("Avg loss (validation): {:.4f}".format(avg_loss_val))
        print("Avg acc (validation): {:.4f}".format(avg_acc_val))
        print('-' * 10)
        print()

        # Update model based on change in accuracy
        if epoch == 0:
            best_acc = avg_acc_val
            best_loss = avg_loss_val
            # save checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_acc,
                'best_loss': best_loss,
                'train_accuracy': train_accuracy,
                'train_losses': train_losses,
                'val_accuracy': val_accuracy,
                'val_losses': val_losses,
                'elapsed_time': elapsed_time,
                'patience_counter': patience_counter
            }, checkpoint)
        else:
            if avg_loss_val < best_loss:
                best_acc = avg_acc_val
                best_loss = avg_loss_val
                # set patience counter to 0
                patience_counter = 0
                # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_accuracy': best_acc,
                    'best_loss': best_loss,
                    'train_accuracy': train_accuracy,
                    'train_losses': train_losses,
                    'val_accuracy': val_accuracy,
                    'val_losses': val_losses,
                    'elapsed_time': elapsed_time,
                    'patience_counter': patience_counter
                }, checkpoint)
                # save best model
                torch.save({
                    model.state_dict(),
                    best_model
                })
            else:
                patience_counter = patience_counter + 1
                print("Loss has not decreased in {} epochs.".format(patience_counter))
                print("Early stopping after {} epochs".format(patience))
                print()

        # Update learning rate
        scheduler.step(avg_loss_val)

        # save checkpoint after every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_accuracy': best_acc,
                'best_loss': best_loss,
                'train_accuracy': train_accuracy,
                'train_losses': train_losses,
                'val_accuracy': val_accuracy,
                'val_losses': val_losses,
                'elapsed_time': elapsed_time,
                'patience_counter': patience_counter
            }, checkpoint)

        # Early stopping
        if patience_counter == patience:
            print('-' * 60)
            print("No decrease in loss after {} epochs. Will stop now.".format(patience))
            print('-' * 60)
            break

    # Capture total time taken
    elapsed_time = elapsed_time + time.time() - since
    print()
    print("Training completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Best acc: {:.4f}".format(best_acc))
    print("Best loss: {:.4f}".format(best_loss))

    # Generate plots for losses
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'bo',
             label='training loss')

    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r',
             label='validation loss')

    plt.legend()
    plt.show()
    plt.close()

    # Generate plots for accuracy
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, 'bo',
             label='train accuracy')

    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, 'r',
             label='val accuracy')

    plt.legend()
    plt.show()
    plt.close()

    print('Done!')
    return None
