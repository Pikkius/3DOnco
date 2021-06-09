import os

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from model import model_3DOnco

import copy
from datetime import datetime

from tqdm import tqdm
from torch.utils.data import DataLoader


def Train(train_set, val_set, config):
    train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    net = model_3DOnco('conv', config.inputs_voc, config.hidden_dim)
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=config.LR,
                          momentum=config.MOMENTUM, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

    loss_list_train = []
    acc_list_train = []
    loss_list_val = []
    acc_list_val = []
    max_accuracy = 0
    current_step = 0

    for epoch in tqdm(range(config.NUM_EPOCHS)):

        tot_train_loss = 0
        running_corrects = 0

        print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, config.NUM_EPOCHS, scheduler.get_lr()))
        net.train()  # Sets module in training mode
        # Iterate over the dataset

        for x in tqdm(train_dataloader):
            # Bring data over the device of choice
            x = [x[i].to(config.DEVICE) for i in range(len(x))]

            # Forward pass to the network
            outputs = net(x[:-1])  # features dim = [batch, vocab, seq_len, (seq)]

            # Compute loss based on output and ground truth
            loss = criterion(outputs, x[-1])

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad()  # Zero-ing the gradients

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == x[-1].data).data.item()

            tot_train_loss += loss.item() * config.BATCH_SIZE

            # Log loss
            if current_step % config.LOG_FREQUENCY == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # Calculate Accuracy
        train_accuracy = running_corrects / float(len(train_set))
        epoch_train_loss = tot_train_loss / len(train_set)
        loss_list_train.append(epoch_train_loss)
        acc_list_train.append(train_accuracy)

        # Calculate Accuracy and loss on val
        val_accuracy, epoch_val_loss = evaluation(net, val_dataloader, criterion=criterion, device=config.DEVICE)

        loss_list_val.append(epoch_val_loss)
        acc_list_val.append(val_accuracy)

        if val_accuracy > max_accuracy:
            best_step = current_step
            max_accuracy = val_accuracy
            best_net = copy.deepcopy(net)

        # Step the scheduler
        scheduler.step(epoch)

    print('Best model found at step {}'.format(best_step))

    if config.save_out:
        if config.out_dir is None:
            dateTimeObj = datetime.now()
            out_dir = dateTimeObj.strftime("%d_%b_%Y_%H)")

            os.mkdir(out_dir)
        else:
            out_dir = config.out_dir

        # np.save(timestampStr'/config.txt', config)

        np.save(out_dir + '/result.txt', [loss_list_train, acc_list_train, loss_list_val, acc_list_val])

    return best_net


def Test(test_set, model, batch):
    test_dataloader = DataLoader(test_set, batch_size=batch, shuffle=False, num_workers=0)

    return evaluation(model, test_dataloader)


def evaluation(model, dataset, criterion=None, device='cpu'):
    model.train(False)  # Set Network to evaluation mode
    running_corrects = 0
    tot_loss = 0

    with torch.no_grad():
        for x in tqdm(dataset):

            x = [x[i].to(device) for i in range(len(x))]

            # Forward Pass
            outputs = model(x[-1])

            # Compute loss based on output and ground truth
            if criterion is not None:
                loss = criterion(outputs, x[-1])

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == x[-1].data).data.item()

            if criterion is not None:
                tot_loss += loss.item() * x[0].size(0)

    # Calculate Accuracy
    accuracy = running_corrects / float(len(dataset))

    if criterion is not None:
        epoch_loss = tot_loss / len(dataset)
        return accuracy, epoch_loss
    else:
        return accuracy
