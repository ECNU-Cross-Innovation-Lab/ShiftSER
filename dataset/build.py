import torch
from torch.utils.data import DataLoader

import pickle
import numpy as np
from dataset.IEMOCAP import IEMOCAP

def build_loader(config, ith_fold=1):
    dataset_train, dataset_test = build_dataset(config, ith_fold=ith_fold)
    dataloader_train = DataLoader(dataset_train, batch_size=config.DATA.BATCH_SIZE)
    dataloader_test = DataLoader(dataset_test, batch_size=config.DATA.BATCH_SIZE)
    return dataloader_train, dataloader_test


def build_dataset(config, ith_fold=1):
    with open(config.DATA.DATA_PATH,'rb') as f:
        DataMap = pickle.load(f)
    if config.DATA.DATASET == 'IEMOCAP':
        dataset_train, dataset_test = IEMOCAP.Partition(ith_fold, DataMap)

    return dataset_train, dataset_test

