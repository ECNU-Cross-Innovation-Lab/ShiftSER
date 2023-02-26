import torch
import torch.nn as nn
from .Augment import TemporalShift, Specaugment


class RNN(nn.Module):

    def __init__(self, dim, length, num_classes=4, shift=False, stride=1, n_div=1, bidirectional=True, drop=0.):
        super().__init__()
        self.specaugment = Specaugment(dim)

        if shift:
            rnn = nn.LSTM(input_size=dim,
                          hidden_size=dim,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True,
                          dropout=drop)
            self.rnn = TemporalShift(rnn, stride, n_div, bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=dim,
                               hidden_size=dim,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True,
                               dropout=drop)

        self.norm = nn.LayerNorm(dim * 2)
        self.head = nn.Linear(dim * 2, num_classes)

    def forward(self, x, length=None):
        """
        x: B L D
        """
        x = self.specaugment(x)
        rnn_output, _ = self.rnn(x)

        x = self.head(torch.mean(self.norm(rnn_output), dim=1))
        return x
