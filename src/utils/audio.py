from pathlib import Path
from warnings import filterwarnings

filterwarnings("ignore", category=UserWarning)

import librosa
import numpy as np
import torch as t
import torchaudio

from utils.audio_augmentation import WhiteNoisePerturbation, ShiftPerturbation

PERTURB = True

shift = ShiftPerturbation()
whitenoise = WhiteNoisePerturbation()


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
    else:
        minval = -120
        maxval = 70
        tensor.add_(-minval)
        tensor.div_(maxval - minval)
        tensor.mul_(2)
        tensor.add_(-1)
    return tensor


def get_mfcc(file: Path, length: int = None, normalization: str = "minmax", training=False):
    window_size = 0.025
    window_stride = 0.01
    values, sampling_rate = librosa.load(file, sr=None)
    assert sampling_rate == 16000

    if PERTURB and training:
        shift.perturb(values)
        whitenoise.perturb(values)

    fft_window_length = int(sampling_rate * window_size)
    hop_length = int(sampling_rate * window_stride)


    mel_kwargs = {}
    mel_kwargs['f_min'] = 0.0
    mel_kwargs['f_max'] = None
    mel_kwargs['n_mels'] = 64

    mel_kwargs['n_fft'] = fft_window_length  # 512

    mel_kwargs['win_length'] = fft_window_length
    mel_kwargs['hop_length'] = hop_length
    mel_kwargs['window_fn'] = t.hann_window

    featurizer = torchaudio.transforms.MFCC(
        sample_rate=sampling_rate,
        n_mfcc=64,
        dct_type=2,
        norm="ortho",
        log_mels=True,
        melkwargs=mel_kwargs
    )

    cepstrogram = featurizer(t.FloatTensor(values))
    image_len = cepstrogram.shape[-1]

    if length is not None:
        pad_left = (length - image_len) // 2
        pad_right = (length - image_len) // 2

        if (length - image_len) % 2 == 1:
            pad_right += 1

        cepstrogram = t.nn.functional.pad(cepstrogram, [pad_left, pad_right], mode="constant", value=0)

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
