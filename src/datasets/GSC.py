import matplotlib.pyplot as plt

from pathlib import Path
from shutil import move

import numpy as np
import torch as t
from torch.utils.data import Dataset

from utils.audio import get_mfcc


class GoogleSpeechCommandDataset(Dataset):
    _core_words = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go",
                   "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

    def __init__(self, root_dir: Path, mode="train"):
        super().__init__()
        assert mode in ["train", "test", "valid"]
        self.root_dir = root_dir

        # If the directory split has not been made
        if not self.root_dir.joinpath("_valid").exists() or not self.root_dir.joinpath("_test").exists():
            for splitdir in ("_valid", "_test"):
                root_dir.joinpath(splitdir).mkdir()
                for word in self._core_words:
                    root_dir.joinpath(splitdir).joinpath(word).mkdir()

            for (listfile, dirname) in [("validation_list.txt", "_valid"), ("testing_list.txt", "_test")]:
                list_path = root_dir.joinpath(listfile)
                new_list = []
                with list_path.open("r") as readfile:
                    lines = readfile.readlines()
                for line in lines:
                    line = line.rstrip()
                    if not line.split("/")[0] in self._core_words:
                        continue
                    move(root_dir.joinpath(line),
                         root_dir.joinpath(dirname).joinpath(line))

        if mode == "test":
            self.root_dir = self.root_dir.joinpath("_test")
        elif mode == "valid":
            self.root_dir = self.root_dir.joinpath("_valid")
        _subdirectories = [subdir for subdir in self.root_dir.iterdir() if subdir.is_dir()]
        self.classes = [subdir for subdir in _subdirectories if subdir.name in self._core_words]

        self.files = [(filename, self.class_str_to_int(category.name)) for category in self.classes for
                      filename in list(category.iterdir())]

    def class_str_to_int(self, category):
        return [x.name for x in self.classes].index(category)

    def __getitem__(self, index):
        filepath, target = self.files[index]
        # sample_rate, samples = wavfile.read(filepath)
        # frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)
        # plt.pcolormesh(times, frequencies, spectrogram)
        cepstrogram = get_mfcc(filepath, length=34)
        return cepstrogram, target

    def __len__(self):
        return len(self.files)
