from pathlib import Path

import librosa
import numpy as np
import torch as t


def get_mfcc(file: Path, length: int = None, normalization="standardization"):
    window_size = 0.025
    window_stride = 0.02
    values, sampling_rate = librosa.load(file, sr=None)
    fft_window_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)

    cepstrogram = librosa.feature.mfcc(values, sampling_rate, n_mfcc=20, hop_length=hop_length, n_fft=fft_window_length)

    if length is not None:
        cepstrogram = cepstrogram[:, :length]
        if cepstrogram.shape[1] < length:
            cepstrogram = np.hstack((cepstrogram, np.zeros((cepstrogram.shape[0], length - cepstrogram.shape[1]))))
    cepstrogram = np.resize(cepstrogram, (1, cepstrogram.shape[0], cepstrogram.shape[1]))
    cepstrogram = t.FloatTensor(cepstrogram)

    if normalization == "standardization":
        cepstrogram.add_(-cepstrogram.mean())
        cepstrogram.div_(cepstrogram.std())
    elif normalization == "minmax":
        minval = cepstrogram.min()
        maxval = cepstrogram.max()
        cepstrogram.add_(-minval)
        cepstrogram.div_(maxval - minval)

    return cepstrogram
