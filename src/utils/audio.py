from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import torch as t


def get_audio(file: Path, length: int = None):
    values, sr = librosa.load(file, sr=None)
    original_length = len(values)
    if length is not None:
        values = values[:length]
        if values.shape[0] < length:
            values = np.hstack((values, np.zeros(length - values.shape[0])))
    return t.tensor(values).to(t.float32), original_length, sr


def _normalize(tensor: t.Tensor, mode: str) -> t.Tensor:
    assert mode in ["standardization", "minmax"]
    if mode == "standardization":
        tensor.add_(-tensor.mean())
        tensor.div_(tensor.std())
    elif mode == "minmax":
        minval = tensor.min()
        maxval = tensor.max()
        tensor.add_(-minval)
        tensor.div_(maxval - minval)
    return tensor


def get_mfcc(file: Path, length: int = None, normalization: str = "standardization"):
    window_size = 0.02
    window_stride = 0.01
    values, sampling_rate = librosa.load(file, sr=None)
    fft_window_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)

    cepstrogram = librosa.feature.mfcc(values, sampling_rate, n_mfcc=64, hop_length=hop_length, n_fft=fft_window_length)

    if length is not None:
        cepstrogram = cepstrogram[:, :length]
        if cepstrogram.shape[1] < length:
            cepstrogram = np.hstack((cepstrogram, np.zeros((cepstrogram.shape[0], length - cepstrogram.shape[1]))))
    cepstrogram = np.resize(cepstrogram, (1, cepstrogram.shape[0], cepstrogram.shape[1]))
    cepstrogram = t.FloatTensor(cepstrogram)

    cepstrogram = _normalize(cepstrogram, normalization)
    return cepstrogram


def get_spectrogram(file: Path, length: int = None, normalization: str = "standardization"):
    values, sampling_rate = librosa.load(file, sr=None)
    spectrogram = librosa.feature.melspectrogram(y=values, sr=sampling_rate)

    if length is not None:
        spectrogram = spectrogram[:, :length]
        if spectrogram.shape[1] < length:
            spectrogram = np.hstack((spectrogram, np.zeros((spectrogram.shape[0], length - spectrogram.shape[1]))))
    spectrogram = np.resize(spectrogram, (1, spectrogram.shape[0], spectrogram.shape[1]))
    spectrogram = t.FloatTensor(spectrogram)

    spectrogram = _normalize(spectrogram, normalization)
    return spectrogram
