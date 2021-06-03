import torch
import math
import collections
import numpy as np


# from data import

SEQ_LEN = 3000


class model_3DOnco(torch.nn.Module):

    def __init__(self, mode, inputs_voc, hidden_dim):
        super(model_3DOnco, self).__init__()
        if mode == 'linear':
            self.seq_feature = [seq_linear(hidden_dim)] * 4
        elif mode == 'conv':
            self.seq_feature = [seq_conv(inputs_voc[i], hidden_dim) for i in range(4)]

        self.seq_feature = self.seq_feature + [torch.nn.Conv2d]
        self.conv_seq = torch.nn.Conv1d(1, 8, padding=2, kernel_size=5, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(8)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=5, padding=2)
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.seq_feature(x)  # [batch, feature, vocab, seq_len]
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dp(out)

        return out


class seq_linear(torch.nn.Module):

    def __init__(self, hidden_dim=8):
        super(seq_linear, self).__init__()
        self.linear = torch.nn.Linear(SEQ_LEN, hidden_dim, bias=True) # batch, aminoacid, vocabulary
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

    def __init__(self, input_voc, hidden_dim=8):
        super(seq_conv, self).__init__()
        self.conv = torch.nn.Conv1d(input_voc, hidden_dim, padding=2, kernel_size=5, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=5, padding=2)
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dp(out)

        return out

