from turtle import forward
import os
import torch
import torchaudio
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class Pretrain_Model(nn.Module):

    def __init__(self, model, pretrain='hubert', finetune=True) -> None:
        super().__init__()
        assert pretrain in ['hubert', 'wav2vec2'], "Unkown pretrain model for finetuning"
        if pretrain == 'hubert':
            bundle = torchaudio.pipelines.HUBERT_BASE
            self.pretrain = bundle.get_model()
        elif pretrain == 'wav2vec2':
            bundle = torchaudio.pipelines.WAV2VEC2_BASE
            self.pretrain = bundle.get_model()
        self.finetune = finetune
        self.model = model

    def forward(self, x, length=None):
        """
        x: B L D
        B for batchsize, L for length, D for dim/channel
        """
        if self.finetune:
            with torch.no_grad():
                x, _ = self.pretrain.feature_extractor(x, None)
            x = self.pretrain.encoder(x, None)
        else:
            self.pretrain.eval()
            with torch.no_grad():
                x,_ = self.pretrain(x, None)
        x = self.model(x)
        return x