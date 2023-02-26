from numpy import short
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from .Augment import TemporalShift, Specaugment
from timm.models.layers import trunc_normal_


class Permute(nn.Module):

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.permute(x, self.dims)


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class selfattention(nn.Module):

    def __init__(self, dim, length, num_heads, position_embedding_type='absolute'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.position_embedding_type = position_embedding_type
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = length
            self.distance_embedding = nn.Embedding(2 * length - 1, self.head_dim)
            # trunc_normal_(self.distance_embedding.weight, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x: B L D
        """
        B, L, D = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, D // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attention_scores = (q @ k.transpose(-2, -1))
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = x.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=x.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=x.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=q.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", q, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", q, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", k, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores * self.scale
        attention_probs = self.softmax(attention_scores)
        output = attention_probs @ v
        output = output.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self,
                 dim,
                 length,
                 num_heads=4,
                 mlp_ratio=4,
                 drop=0.,
                 position_embedding_type='relative_key',
                 shift=False,
                 stride=1,
                 n_div=1,
                 bidirectional=False):
        super().__init__()

        ##Transformer with shift
        # if shift:
        #     self.shift = TemporalShift(nn.Identity(), stride, n_div, bidirectional)
        # else:
        #     self.shift = None

        ##Shiftformer
        self.shift = shift
        if self.shift:
            self.token_mixer = TemporalShift(nn.Identity(), stride, n_div, bidirectional)
        else:
            self.token_mixer = selfattention(dim, length, num_heads, position_embedding_type)
            self.norm1 = nn.LayerNorm(dim)

        self.mlp = Mlp(dim, dim * mlp_ratio, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        ##Transformer with residual-shift
        # if self.shift:
        #     shortcut = self.shift(x)
        # else:
        #     shortcut = x
        ##Transformer with in-place-shift
        # if self.shift:
        #     x = self.shift(x)
        # shortcut = x
        # x = self.attention(x)
        # x = self.norm1(shortcut + x)
        # shortcut = x
        # x = self.mlp(x)
        # x = self.norm2(shortcut + x)

        ##Shiftformer
        if self.shift:
            x = self.token_mixer(x)
        else:
            shortcut = x
            x = self.token_mixer(x)
            x = self.norm1(shortcut + x)
        shortcut = x
        x = self.mlp(x)
        x = self.norm2(shortcut + x)

        return x


class Transformer(nn.Module):

    def __init__(self,
                 dim,
                 length,
                 num_classes=4,
                 shift=False,
                 stride=1,
                 n_div=1,
                 bidirectional=False,
                 mlp_ratio=4,
                 drop=0.,
                 position_embedding_type='relative_key_query'):
        super().__init__()

        assert position_embedding_type in ['relative_key_query', 'absolute',
                                           'relative_key'], "Unknown position embedding"

        self.specaugment = Specaugment(dim)

        self.position_embedding_type = position_embedding_type
        if position_embedding_type == 'absolute':
            self.position_embeddings = nn.Embedding(length, dim)
            # trunc_normal_(self.position_embeddings.weight, std=.02)
            self.register_buffer("position_ids", torch.arange(length).expand((1, -1)))

        num_heads = 12
        if shift:
            shift = [True, True]
        else:
            shift = [False, False]

        self.transformer = nn.Sequential(*[
            TransformerEncoder(dim, length, num_heads, mlp_ratio, drop, position_embedding_type, shift[i], stride,
                               n_div, bidirectional) for i in range(2)
        ])
        self.head = nn.Linear(dim, num_classes)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x, length=None):
        """
        x: B L D
        """
        x = self.specaugment(x)
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(self.position_ids)
            x = x + position_embeddings
        x = self.transformer(x)
        x = self.head(torch.mean(x, dim=1))
        return x