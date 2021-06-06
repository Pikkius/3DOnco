import torch
import math
import collections
import numpy as np


# from data import

SEQ_LEN = 3000


class model_3DOnco(torch.nn.Module):

    def __init__(self, mode, inputs_voc, hidden_dim):
        super(model_3DOnco, self).__init__()
        self.mode = mode
        if mode == 'linear':
            self.seq_feature = [seq_linear(hidden_dim)] * 4
        elif mode == 'conv':
            self.seq_feature = [seq_conv(inputs_voc[i], hidden_dim) for i in range(4)]

        # [batch, bins, seq, seq]
        self.dist_feature = torch.nn.Sequential(
            torch.nn.Conv2d(inputs_voc, hidden_dim*2, stride=4, padding=2, kernel_size=7, bias=False),
            torch.nn.BatchNorm2d(hidden_dim*2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=2),  # [batch, bins, seq, seq]
            torch.nn.Dropout()
        )

        # non so la dimensione (da stampare)
        self.dist_linear_2d = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, hidden_dim*4),  # [batch, seq, seq, bins]
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )
        self.dist_linear_1d = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )
        self.seq_linear = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim*4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.BatchNorm1d(hhidden_dim * 4),
            torch.nn.Linear(hidden_dim * 4, hidden_dim * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 4, 1)
        )

        self.maxpool = torch.nn.MaxPool1d(kernel_size=5, padding=2)

    def forward(self, x):
        out_seq = self.seq_feature(x[:, :-1])  # [batch, feature, seq_len, vocab]
        out_seq = self.seq_linear(out_seq)

        out_dist = self.dist_feature(x[:, -1])
        print(out_dist.shape)
        out_dist = self.dist_linear_2d(out_dist)
        out_dist = out_dist.view(out_dist.size(0), -1)
        print(out_dist.shape)
        out_dist = self.dist_linear_1d(out_dist)

        # reunion
        out = np.append(out_seq, out_dist.unsqueeze(1))  # [batch, feature, seq_len, vocab]
        out = self.classifier(out)

        return out


class seq_linear(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super(seq_linear, self).__init__()
        self.linear = torch.nn.Linear(SEQ_LEN, hidden_dim, bias=True)  # batch, aminoacid, vocabulary
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.linear(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dp(out)

        return out


class seq_conv(torch.nn.Module):

    def __init__(self, inputs_voc, hidden_dim=8):
        super(seq_conv, self).__init__()
        self.conv = torch.nn.Conv1d(inputs_voc, hidden_dim, stride=2, padding=2, kernel_size=5, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=5, padding=2)
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv(x)  # da fare la transposta perch conv lavora sulla seconda dimensione
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dp(out)

        return out
