import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split

from tqdm import tqdm

from data import get_labels, Protein
from model import model_3DOnco

if __name__ == '__main__':
    dataset = Protein(root='.data')

    labels = get_labels(root='.data')
    lines = np.arange(len(labels))
    train_tmp_indexes, test_indexes, label_train_tmp, label_test = train_test_split(lines, labels,
                                                                                    test_size=0.1, stratify=labels)
    train_indexes, val_indexes, label_train, label_val = train_test_split(train_tmp_indexes, label_train_tmp,
                                                                          test_size=0.2, stratify=label_train_tmp)

    train_dataset = Subset(dataset, train_indexes)
    val_dataset = Subset(dataset, val_indexes)
    test_dataset = Subset(dataset, test_indexes)
    # Check dataset sizes
    print('Train Dataset: {}'.format(len(train_dataset)))
    print('Valid Dataset: {}'.format(len(val_dataset)))
    print('Test Dataset: {}'.format(len(test_dataset)))

    DEVICE = 'cpu'
    BATCH_SIZE = 1

    # Dataloaders iterate over pytorch datasets and transparently provide useful functions (e.g. parallelization and shuffling)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    res = dataset.__getitem__(1)

    inputs_voc = [20, 9, 37, 37, 10]

    LR = 0.007  # The initial Learning Rate
    MOMENTUM = 0.9  # Hyperparameter for SGD, keep this at 0.9 when using SGD
    WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default
    NUM_EPOCHS = 5  # Total number of training epochs (iterations over dataset)
    STEP_SIZE = 20
    GAMMA = 0.1
    LOG_FREQUENCY = 1

    net = model_3DOnco('conv', inputs_voc, 8)
    criterion = nn.CrossEntropyLoss()
    parameters_to_optimize = net.parameters()
    optimizer = optim.SGD(parameters_to_optimize, lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    for epoch in range(NUM_EPOCHS):
        logs = {}
        tot_train_loss = 0
        tot_val_loss = 0
        running_corrects = 0
        current_step = 0
        print('Starting epoch {}/{}, LR = {}'.format(epoch + 1, NUM_EPOCHS, scheduler.get_lr()))
        net.train()  # Sets module in training mode
        # Iterate over the dataset

        for seq, ss, phi, psi, matrix, labels in train_dataloader:
            # Bring data over the device of choice

            seq = seq.to(DEVICE)
            ss = ss.to(DEVICE)
            phi = phi.to(DEVICE)
            psi = psi.to(DEVICE)
            matrix = matrix.to(DEVICE)
            labels = labels.to(DEVICE)

            #

            # Forward pass to the network
            outputs = net([seq, ss, phi, psi, matrix])  # features dim = [batch, vocab, seq_len, (seq)]

            # Compute loss based on output and ground truth
            loss = criterion(outputs, labels)

            # PyTorch, by default, accumulates gradients after each backward pass
            # We need to manually set the gradients to zero before starting a new iteration
            optimizer.zero_grad()  # Zero-ing the gradients

            # Get predictions
            _, preds = torch.max(outputs.data, 1)

            # Update Corrects
            running_corrects += torch.sum(preds == labels.data).data.item()

            tot_train_loss += loss.item() * images.size(0)

            # Log loss
            if current_step % LOG_FREQUENCY == 0:
                print('Step {}, Loss {}'.format(current_step, loss.item()))

            # Compute gradients for each layer and update weights
            loss.backward()  # backward pass: computes gradients
            optimizer.step()  # update weights based on accumulated gradients

            current_step += 1

        # Calculate Accuracy
        '''train_accuracy = running_corrects / float(len(train_dataset))
        epoch_train_loss = tot_train_loss / len(train_dataset)
        loss_list_train.append(epoch_train_loss)
        acc_list_train.append(train_accuracy)'''

        net.train(False)  # Set Network to evaluation mode
        running_corrects = 0

        with torch.no_grad():
            for images, labels in tqdm(val_dataloader):
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                # Forward Pass
                outputs = net(images)

                # Compute loss based on output and ground truth
                loss = criterion(outputs, labels)

                # Get predictions
                _, preds = torch.max(outputs.data, 1)

                # Update Corrects
                running_corrects += torch.sum(preds == labels.data).data.item()

                tot_val_loss += loss.item() * images.size(0)

        # Calculate Accuracy
        '''val_accuracy = running_corrects / float(len(val_dataset))
        epoch_val_loss = tot_val_loss / len(val_dataset)
        loss_list_val.append(epoch_val_loss)
        acc_list_val.append(val_accuracy)

        if (val_accuracy > max_accuracy):
            max_accuracy = val_accuracy
            best_net = copy.deepcopy(net)'''

        # Step the scheduler
        scheduler.step()
