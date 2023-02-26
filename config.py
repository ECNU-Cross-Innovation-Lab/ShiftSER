import os
from pickle import FALSE
from yacs.config import CfgNode as CN

dataset = {
    'IEMOCAP': {
        'wav_path': './dataset/IEMOCAP/wavfeature_7.5.pkl',
        'length': 374,  # the length of pretrained representation when the input is 7.5s and sampled at 16kHZ
        'num_classes': 4,
        'num_fold': 5
    }
}

_C = CN()

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
# Path of log
_C.LOGPATH = './log'
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, overwritten by funcition ConfigDataset
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'IEMOCAP'
# Feature augmentation
_C.DATA.SPEAUG = True
# Input channel
_C.DATA.DIM = 768
# Sequence Length of pretrained representation
_C.DATA.LENGTH = 374
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type :['rnn', 'transformer','cnn']
_C.MODEL.TYPE = 'cnn'
# Model name, auto-renamed later
_C.MODEL.NAME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 4
# Pretrained Model in ['hubert','wav2vec2']
_C.MODEL.PRETRAIN = 'wav2vec2'
# Dropout rate
_C.MODEL.DROP_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0
# Whether to use temporal shift
_C.MODEL.USE_SHIFT = False
# kernel size of convolution
_C.MODEL.KERNEL_SIZE = 7
# path to save model
_C.MODEL.SAVE_PATH = ''
# whether to save model
_C.MODEL.SAVE = False

# Transformer parameters
_C.MODEL.Trans = CN()
_C.MODEL.Trans.POSITION = 'relative_key_query'
_C.MODEL.Trans.MLP_RATIO = 4

# Temporal Shift parameters
_C.MODEL.SHIFT = CN()
_C.MODEL.SHIFT.STRIDE = 1
_C.MODEL.SHIFT.N_DIV = 4
_C.MODEL.SHIFT.BIDIRECTIONAL = False
_C.MODEL.SHIFT.PADDING = 'zero'

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 100
_C.TRAIN.WARMUP_EPOCHS = 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 10
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# whether to finetune or feature extraction
_C.TRAIN.FINETUNE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
_C.SEED = 42
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# fold validation
_C.NUM_FOLD = 5


def ConfigDataset(config):
    config.defrost()
    config.DATA.DATA_PATH = dataset[config.DATA.DATASET]['wav_path']
    config.DATA.LENGTH = dataset[config.DATA.DATASET]['length']
    config.MODEL.NUM_CLASSES = dataset[config.DATA.DATASET]['num_classes']
    config.NUM_FOLD = dataset[config.DATA.DATASET]['num_fold']
    config.freeze()


def ConfigPretrain(config):
    config.defrost()
    if config.TRAIN.FINETUNE:
        # use wav2vec2 for finetune
        config.MODEL.PRETRAIN = 'wav2vec2'
        config.TRAIN.OPTIMIZER.NAME = 'adam'
        config.TRAIN.LR_SCHEDULER.NAME = 'lambda'
    else:
        # use hubert for feature extraction
        config.MODEL.PRETRAIN = 'hubert'
        config.TRAIN.OPTIMIZER.NAME = 'adamw'
        config.TRAIN.LR_SCHEDULER.NAME = 'cosine'
    config.freeze()


def Update(config, args):
    config.defrost()

    if args.batchsize:
        config.DATA.BATCH_SIZE = args.batchsize
    if args.model:
        config.MODEL.TYPE = args.model
    if args.shift:
        config.MODEL.USE_SHIFT = True
    if args.stride:
        config.MODEL.SHIFT.STRIDE = args.stride
    if args.ndiv:
        config.MODEL.SHIFT.N_DIV = args.ndiv
    if args.bidirectional:
        config.MODEL.SHIFT.BIDIRECTIONAL = True
    if args.finetune:
        config.TRAIN.FINETUNE = True
    if args.gpu:
        config.LOCAL_RANK = int(args.gpu)
    if args.seed:
        config.SEED = args.seed

    config.freeze()


def Rename(config):
    config.defrost()
    if config.MODEL.NAME == '':
        config.MODEL.NAME = config.MODEL.TYPE
    if config.TRAIN.FINETUNE:
        config.MODEL.NAME = config.MODEL.NAME + '_finetune' + config.MODEL.PRETRAIN
    else:
        config.MODEL.NAME = config.MODEL.NAME + '_featurex' + config.MODEL.PRETRAIN
    if config.MODEL.USE_SHIFT:
        config.MODEL.NAME = config.MODEL.NAME + '+shift' + str(config.MODEL.SHIFT.N_DIV)
        if config.MODEL.SHIFT.BIDIRECTIONAL:
            config.MODEL.NAME = config.MODEL.NAME + 'b'
        config.MODEL.NAME = config.MODEL.NAME + 'stride' + str(config.MODEL.SHIFT.STRIDE)

    config.MODEL.NAME = config.MODEL.NAME + '-' + config.DATA.DATA_PATH.split('/')[-1].split('.pkl')[0]
    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    config = _C.clone()
    ConfigDataset(config)
    Update(config, args)
    ConfigPretrain(config)
    Rename(config)
    return config
