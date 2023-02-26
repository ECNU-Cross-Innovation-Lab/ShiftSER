import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.wav2vec2.modeling_wav2vec2 import _compute_mask_indices


class TemporalShift(nn.Module):

    def __init__(self, net, stride=1, n_div=8, bidirectional=False, padding='zero') -> None:
        """
        net: the input of net module will be shifted
        stride: the number of steps to be shifted, we adopt stride=1 in our paper
        n_div: reciprocal of proportion of shift, namely 1/n_div of channels or features will be shifted
        bidirectional: whether to aodpt bidirectional temporal shift
        padding: use 'zero' or 'repeat' to pad the last/first frame feature, we adopt 'zero' padding in our paper
        """
        super().__init__()
        self.net = net
        self.stride = stride
        self.n_div = n_div
        self.bidirectional = bidirectional
        assert padding in ['zero', 'repeat'], "Unkown shift type"
        self.padding = padding

    def forward(self, x):
        """
        x: B L D
        shape of returned tensor is same as original output of net
        """
        x = self.shift(x, self.stride, self.n_div, self.bidirectional, self.padding)
        if isinstance(self.net, nn.Conv1d):
            x = x.transpose(-2, -1)
        return self.net(x)

    @staticmethod
    def shift(x, stride, n_div, bidirectional, padding):
        B, L, D = x.size()
        fold = D // n_div  # number of fetures to be shifted
        if padding == 'zero':
            out = torch.zeros_like(x)
            if bidirectional:
                out[:, stride:, :fold] = x[:, :-stride, :fold]
                out[:, :-stride, fold:2 * fold] = x[:, stride:, fold:2 * fold]
                if n_div != 2:
                    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]
            else:
                out[:, stride:, :fold] = x[:, :-stride, :fold]
                out[:, :, fold:] = x[:, :, fold:]
        elif padding == 'repeat':
            out = x
            if bidirectional:
                out[:, stride:, :fold] = out[:, :-stride, :fold]
                out[:, :-stride, fold:2 * fold] = out[:, stride:, fold:2 * fold]
            else:
                out[:, stride:, :fold] = out[:, :-stride, :fold]
        return out


class Specaugment(nn.Module):
    """
    For convenience, we apply specaugment to the output of pretrained model.
    The code reference from https://github.com/b04901014/FT-w2v2-ser/blob/main/modules/FeatureFuser.py
    """

    def __init__(self, dim) -> None:
        super().__init__()
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(dim).uniform_())
        self.mask_time_length = 15
        self.mask_time_prob = 0.08
        self.observe_time_prob = 0.0
        self.mask_feature_length = 64
        self.mask_feature_prob = 0.05

    def forward(self, x):
        """
        x: B L D
        """
        if not self.training:
            return x
        batch_size, sequence_length, hidden_size = x.size()
        if self.mask_time_prob > 0:
            mask_time_indices = _compute_mask_indices((batch_size, sequence_length),
                                                      self.mask_time_prob,
                                                      self.mask_time_length,
                                                      min_masks=2)
            mask_time_indices = torch.tensor(mask_time_indices, device=x.device, dtype=torch.bool)
            flip_mask = torch.rand((batch_size, sequence_length)) > self.observe_time_prob
            x[mask_time_indices & flip_mask.cuda()] = self.masked_spec_embed.to(x.dtype)
        if self.mask_feature_prob > 0:
            mask_feature_indices = _compute_mask_indices((batch_size, hidden_size),
                                                         self.mask_feature_prob,
                                                         self.mask_feature_length,
                                                         min_masks=1)
            mask_feature_indices = torch.tensor(mask_feature_indices, device=x.device, dtype=torch.bool)
            x[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        return x