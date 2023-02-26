# The code architectue is adopted from https://github.com/microsoft/Swin-Transformer
import os
import sys
import logging
import argparse
import numpy as np
from time import strftime, localtime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix

import torch
import torch.backends.cudnn as cudnn
from timm.loss import LabelSmoothingCrossEntropy

from config import get_config
from models import build_model
from dataset import build_loader
from optimizer import build_optimizer
from lr_scheduler import build_scheduler
from metrics import IEMOCAP_Meter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_option():
    parser = argparse.ArgumentParser('ShiftSER', add_help=False)

    # # easy config modification
    # parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    # parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--batchsize',type=int, help="batch size for single GPU")
    parser.add_argument('--model', type=str, choices=['cnn', 'rnn', 'transformer'], help='model type')
    parser.add_argument('--shift', action='store_true', help='whether to use temporal shift')
    parser.add_argument('--stride', type=int, help='temporal shift stride')
    parser.add_argument('--ndiv', type=int, help='temporal shift portion (1/ndiv of feautures will be shifted)')
    parser.add_argument('--bidirectional', action='store_true', help='temporal shift direction')
    parser.add_argument('--finetune', action='store_true', help='whether to finetune or feature extraction')
    parser.add_argument('--gpu', type=str, help='gpu rank to use')
    parser.add_argument('--seed', type=int, help='seed')

    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return config


def main(config):
    result = []
    for ith_fold in range(1, config.NUM_FOLD + 1):
        WA, UA = solve(config, ith_fold)
        result.append([WA, UA])
    logger.info('#' * 30 + f'  Summary  ' + '#' * 30)
    logger.info('fold\tWA\tUA\t')
    for ith_fold in range(1, config.NUM_FOLD + 1):
        WA, UA = result[ith_fold - 1]
        logger.info(f'{ith_fold}\t{WA:.4f}\t{UA:.4f}')
    result = np.array(result)
    WA_mean = np.mean(result[:, 0])
    UA_mean = np.mean(result[:, 1])
    logger.info('Avg_WA\tAvg_UA')
    logger.info(f'{WA_mean}\t{UA_mean}')


def solve(config, ith_fold):
    dataloader_train, dataloader_test = build_loader(config, ith_fold=ith_fold)
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")

    optimizer = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(dataloader_train))

    if config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    logger.info('#' * 30 + '  Start Training  ' + '#' * 30)
    Meter = IEMOCAP_Meter()

    for epoch in range(config.TRAIN.EPOCHS):
        logger.info(f'>> epoch {epoch}')
        train_loss = train_one_epoch(config, model, criterion, dataloader_train, optimizer, epoch, lr_scheduler)

        test_loss, WA, UA, pred, label = validate(config, dataloader_test, model)
        logger.info(f'train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, WA: {WA:.4f}, UA: {UA:.4f}')
        # if Meter.UA < UA and config.MODEL.SAVE:
        #     torch.save(model.state_dict(), f'{config.MODEL.SAVE_PATH}/shiftformer{ith_fold}.pth')
        Meter.update(WA, UA, pred, label)
    logger.info('#' * 30 + f'  Summary fold{ith_fold}  ' + '#' * 30)
    logger.info(f'MAX_WA: {Meter.WA:.4f}')
    logger.info('happy\tangry\tsad\tneutral')
    logger.info(confusion_matrix(Meter.label_WA, Meter.pred_WA))
    logger.info(f'MAX_UA: {Meter.UA:.4f}')
    logger.info('happy\tangry\tsad\tneutral')
    logger.info(confusion_matrix(Meter.label_UA, Meter.pred_UA))
    return Meter.WA, Meter.UA


def train_one_epoch(config,
                    model,
                    criterion,
                    dataloader,
                    optimizer,
                    epoch,
                    lr_scheduler):
    total_loss = 0
    optimizer.zero_grad()
    for idx, (samples, length, targets) in enumerate(dataloader):
        samples = samples.cuda()
        length = length.cuda()
        targets = targets.cuda()
        model.train()
        outputs = model(samples, length)
        num_steps = len(dataloader)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        optimizer.step()
        if not config.TRAIN.FINETUNE:
            lr_scheduler.step_update(epoch * num_steps + idx)
    if config.TRAIN.FINETUNE:
        lr_scheduler.step()
    return total_loss


@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0
    pred_list = []
    label_list = []
    for idx, (samples, length, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        length = length.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        output = model(samples, length)
        # measure accuracy and record loss
        loss = criterion(output, targets)
        total_loss += loss.item()
        pred = list(torch.argmax(output, 1).cpu().numpy())
        targets = list(targets.cpu().numpy())
        pred_list.extend(pred)
        label_list.extend(targets)
    WA = accuracy_score(label_list, pred_list)
    UA = balanced_accuracy_score(label_list, pred_list)
    # logger.info('happy\tangry\tsad\tneutral')
    # logger.info(confusion_matrix(label_list, pred_list))
    return total_loss, WA, UA, pred_list, label_list


if __name__ == '__main__':
    config = parse_option()
    log_file = '{}-{}-{}.log'.format(config.MODEL.NAME, config.DATA.DATASET, strftime("%Y-%m-%d_%H:%M:%S", localtime()))
    if not os.path.exists(config.LOGPATH):
        os.mkdir(config.LOGPATH)
    logger.addHandler(logging.FileHandler("%s/%s" % (config.LOGPATH, log_file)))
    logger.info('#' * 30 + '  Training Arguments  ' + '#' * 30)
    logger.info(config.dump())
    torch.cuda.set_device(config.LOCAL_RANK)
    seed = config.SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = True
    main(config)
