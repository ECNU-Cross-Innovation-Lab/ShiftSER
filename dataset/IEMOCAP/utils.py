import torchaudio
import torch

def pad_cut(wav, target_length):
    n_frames = wav.shape[1]
    p = target_length - n_frames
    if p > 0:
        wav = torch.nn.functional.pad(wav, (0, int(p)))
    elif p < 0:
        wav = wav[:, 0:int(target_length)]
    return wav


def load_wav(path, second_length=7.5):
    wav, sample_rate = torchaudio.load(path) # sample rate for IEMOCAP (16kHz)
    length = wav.shape[1]
    target_length = sample_rate * second_length
    return pad_cut(wav, target_length).squeeze(0), length

    


