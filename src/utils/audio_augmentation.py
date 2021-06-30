import random
import numpy as np


class ShiftPerturbation:
    def __init__(self, min_shift_ms=-5.0, max_shift_ms=5.0, rng=None):
        self._min_shift_ms = min_shift_ms
        self._max_shift_ms = max_shift_ms
        self._rng = random.Random() if rng is None else rng

    def perturb(self, data):
        shift_ms = self._rng.uniform(self._min_shift_ms, self._max_shift_ms)
        if abs(shift_ms) / 1000 > (len(data) / 16000):
            # TODO: do something smarter than just ignore this condition
            return
        shift_samples = int(shift_ms * 16000 // 1000)
        # logging.debug("shift: %s", shift_samples)
        if shift_samples < 0:
            data[-shift_samples:] = data[:shift_samples]
            data[:-shift_samples] = 0
        elif shift_samples > 0:
            data[:-shift_samples] = data[shift_samples:]
            data[-shift_samples:] = 0


class WhiteNoisePerturbation:
    def __init__(self, min_level=-90, max_level=-46, rng=None):
        self.min_level = int(min_level)
        self.max_level = int(max_level)
        self._rng = np.random.RandomState() if rng is None else rng

    def perturb(self, data):
        noise_level_db = self._rng.randint(self.min_level, self.max_level, dtype='int32')
        noise_signal = self._rng.randn(data.shape[0]) * (10.0 ** (noise_level_db / 20.0))
        data += noise_signal
