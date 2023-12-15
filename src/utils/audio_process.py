import numpy as np


def normalize_wav_np(wav, eps=1e-8, std=None):
    mean = np.mean(wav, axis=-1, keepdims=True)
    if std is None:
        std = np.std(wav, axis=-1, keepdims=True)
    return (wav - mean) / (std + eps)

def normalize_wav(wav, eps=1e-8, std=None):
    mean = wav.mean(-1, keepdim=True)
    if std is None:
        std = wav.std(-1, keepdim=True)
    return (wav - mean) / (std + eps)