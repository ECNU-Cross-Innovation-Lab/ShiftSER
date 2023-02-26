from .rnn import RNN
from .transformer import Transformer
from .convolution import Convolution
from .pretrain import Pretrain_Model


def build_model(config):
    model_type = config.MODEL.TYPE
    finetune = config.TRAIN.FINETUNE
    if model_type == 'transformer':
        model = Transformer(dim=config.DATA.DIM,
                            length=config.DATA.LENGTH,
                            num_classes=config.MODEL.NUM_CLASSES,
                            shift=config.MODEL.USE_SHIFT,
                            stride=config.MODEL.SHIFT.STRIDE,
                            n_div=config.MODEL.SHIFT.N_DIV,
                            bidirectional=config.MODEL.SHIFT.BIDIRECTIONAL,
                            mlp_ratio=config.MODEL.Trans.MLP_RATIO,
                            drop=config.MODEL.DROP_RATE,
                            position_embedding_type=config.MODEL.Trans.POSITION)
    elif model_type == 'cnn':
        model = Convolution(dim=config.DATA.DIM,
                            length=config.DATA.LENGTH,
                            num_classes=config.MODEL.NUM_CLASSES,
                            kernel_size=config.MODEL.KERNEL_SIZE,
                            shift=config.MODEL.USE_SHIFT,
                            stride=config.MODEL.SHIFT.STRIDE,
                            n_div=config.MODEL.SHIFT.N_DIV,
                            bidirectional=config.MODEL.SHIFT.BIDIRECTIONAL,
                            drop=config.MODEL.DROP_RATE)
    elif model_type == 'rnn':
        model = RNN(dim=config.DATA.DIM,
                    length=config.DATA.LENGTH,
                    num_classes=config.MODEL.NUM_CLASSES,
                    shift=config.MODEL.USE_SHIFT,
                    stride=config.MODEL.SHIFT.STRIDE,
                    n_div=config.MODEL.SHIFT.N_DIV,
                    bidirectional=config.MODEL.SHIFT.BIDIRECTIONAL,
                    drop=config.MODEL.DROP_RATE)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    model = Pretrain_Model(model, config.MODEL.PRETRAIN, finetune)
    return model