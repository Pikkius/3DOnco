import torch


def conv_out_len(W, K, P, S):
    return int(((W - K + 2 * P) / S) + 1)


class model_3DOnco(torch.nn.Module):

    def __init__(self, hidden_dim, seq_len, mode=None, inputs_voc=None):
        super(model_3DOnco, self).__init__()
        self.mode = mode
        if mode is not None:
            if inputs_voc is None:
                raise ValueError("input_voc cannot be None")
            if mode == 'linear':
                self.seq_feature = seq_linear(inputs_voc[0], seq_len, hidden_dim)
            elif mode == 'conv':
                self.seq_feature = seq_conv(inputs_voc[0], hidden_dim)

        # [batch, bins, seq, seq]
        # [(W−K+2P)/S]+1
        # W = 3000, K = 7, P = 2, S = 4
        self.dist_feature = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_dim * 2, stride=4, padding=2, kernel_size=7, bias=False),
            # [batch, hidden_dim*2, seq*, seq*
            torch.nn.BatchNorm2d(hidden_dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=5, stride=2, padding=2),  # [batch, bins*, seq*, seq*]
            torch.nn.Dropout()
        )
        out_conv = conv_out_len(W=conv_out_len(W=seq_len, K=7, P=2, S=4),
                                K=5, S=2, P=2)
        # non so la dimensione (da stampare)
        self.dist_linear_2d = torch.nn.Sequential(
            torch.nn.Linear(out_conv, hidden_dim * 4),  # [batch, seq, seq, bins]
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )
        self.dist_linear_1d = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2 * hidden_dim * 4 * hidden_dim * 4, hidden_dim * 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )
        self.seq_linear = torch.nn.Sequential(
            torch.nn.Linear(out_conv, hidden_dim * 8),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout()
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 8, hidden_dim * 4),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(hidden_dim * 4, 2),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        # [feature, batch, vocab, seq_len]
        if self.mode is not None:
            out_seq = self.seq_feature(x)

            out_seq = out_seq.transpose(1, 0)  # [batch, feature, seq_len, vocab]
            out_seq = self.seq_linear(out_seq)

        out_dist = self.dist_feature(x[-1].view(x[-1].size(0),1,x[-1].size(-1), x[-1].size(-1)))  # [batch, vocab, seq, seq]
        out_dist = self.dist_linear_2d(out_dist)  # [batch, vocab, seq, seq]
        out_dist = out_dist.view(out_dist.size(0), -1)  # [batch, seq * seq * vacab]
        out_dist = self.dist_linear_1d(out_dist)

        # reunion
        if self.mode is None:
            out = out_dist.unsqueeze(1)  # [batch, feature, seq_len, vocab]

        else:
            out = torch.cat([out_seq, out_dist.unsqueeze(1)], dim=1)  # [batch, feature, seq_len, vocab]

        out = self.classifier(out.view(out_dist.size(0), -1))

        return out


# sistemare il linear
class seq_linear(torch.nn.Module):

    def __init__(self, input_voc, seq_len, hidden_dim=8):
        super(seq_linear, self).__init__()
        self.linear = torch.nn.Linear(input_voc, hidden_dim, bias=True)  # batch, vocabulary, aminoacid
        self.bn1 = torch.nn.BatchNorm1d(seq_len)
        self.relu = torch.nn.ReLU(inplace=True),
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.linear(x.transpose(-1, -2))
        out = self.bn1(out)
        out = self.relu(out)
        out = self.max(out)
        out = self.dp(out)

        return out


class seq_conv(torch.nn.Module):

    def __init__(self, inputs_voc, hidden_dim=8):
        super(seq_conv, self).__init__()
        self.conv = torch.nn.Conv1d(inputs_voc, hidden_dim * 2, stride=4, padding=2, kernel_size=7, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim * 2)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
        self.dp = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.conv(x)  # da fare la transposta perch conv lavora sulla seconda dimensione
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.dp(out)

        return out
