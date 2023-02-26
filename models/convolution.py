from turtle import forward
from typing import List
import torch
import torch.nn as nn
import numpy as np
from .Augment import TemporalShift,Specaugment


class Permute(nn.Module):
    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class Block(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=7,
                 shift=False,
                 stride=1,
                 n_div=1,
                 bidirectional=False,
                 drop=0.) -> None:
        super().__init__()

        depth_conv = nn.Conv1d(dim, dim, kernel_size, 1, kernel_size // 2, groups=dim)
        point_conv1 = nn.Conv1d(dim, dim * 4, 1)
        point_conv2 = nn.Conv1d(dim * 4, dim, 1)
        self.sequential = nn.Sequential(depth_conv, Permute([0, 2, 1]), nn.LayerNorm(dim), Permute([0, 2, 1]),
                                        point_conv1, nn.GELU(), point_conv2)
        
        if shift:
            self.shift = TemporalShift(nn.Identity(), stride, n_div, bidirectional)
            self.mode = 'residual' # choose in ['residual','inplace']
        else:
            self.shift = None
            self.mode = ''


    def forward(self, x):
        """
        x: B D L
        return: B D L
        """
        
        if self.shift is None or self.mode == 'residual': # Residual Shift
            out = self.sequential(x)
        if self.shift is not None:
            if isinstance(self.shift.net, nn.Conv1d):
                x = self.shift(x.transpose(-2, -1))
            elif isinstance(self.shift.net, nn.Identity):
                x = self.shift(x.transpose(-2, -1)).transpose(-2, -1)
        if self.shift is not None and self.mode == 'inplace':  # In-place Shift
            out = self.sequential(x)

        return x + out


class Convolution(nn.Module):
    def __init__(self,
                 dim,
                 length,
                 num_classes=4,
                 kernel_size=7,
                 shift=False,
                 stride=1,
                 n_div=1,
                 bidirectional=True,
                 drop=0.):
        super().__init__()

        self.specaugment = Specaugment(dim)

        if shift:
            shift = [False, True]
        else:
            shift = [False, False]
        self.conv_block = nn.Sequential(
            *[Block(dim, kernel_size, shift[i], stride, n_div, bidirectional, drop) for i in range(2)])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)


    def forward(self, x, length=None):
        """
        x: B L D
        """
        x = self.specaugment(x)
        x = x.transpose(-2, -1)  # B D L
        conv_x = self.conv_block(x)
        conv_x = self.norm(conv_x.transpose(-2, -1)).transpose(-2, -1)
        output = self.head(torch.mean(conv_x, dim=-1))
        return output
