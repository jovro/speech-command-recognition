from os import environ
from pathlib import Path
import random

import torch as t
import torch.nn as nn
import torch.optim as optim

from agents.base import BaseAgent
from torch.utils.data import DataLoader
from datasets.GSC import GoogleSpeechCommandDataset
from utils.spectral_augmentation import SpecAugment, SpecCutout
from optimization.Novograd import Novograd
from optimization.PolynomialDecayAnnealing import PolynomialHoldDecayAnnealing
from models.BasicConvNet import BasicNet
from models.EdgeSpeechNets import EdgeSpeechNetA, EdgeSpeechNetB, EdgeSpeechNetC, EdgeSpeechNetD
from models.MatchBoxNet import MatchBoxNet
from models.QuadraticSelfONN import QuadraticSelfONNNet


class GSCAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        self.model = EdgeSpeechNetA(30)  # QuadraticSelfONNNet(10, 3)
        if self.cuda:
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

        self.loss = nn.CrossEntropyLoss()
        self.loss = self.loss.to(self.device)

        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)
        self.optimizer = Novograd(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = PolynomialHoldDecayAnnealing(self.optimizer, max_steps=self.config.epochs, hold_ratio=0.45)
        self.cutout = SpecCutout(rect_masks=5, rect_time=25, rect_freq=15)
        self.augment = SpecAugment(freq_masks=2, time_masks=2, freq_width=15, time_width=25)

        self.dataloader_train = DataLoader(GoogleSpeechCommandDataset(
            Path(environ["GSCPATH"]), mode="train"),
            batch_size=self.config.batch_size, shuffle=True, pin_memory=self.cuda, sampler=None)
        self.dataloader_test = DataLoader(GoogleSpeechCommandDataset(
            Path(environ["GSCPATH"]), mode="test"),
            batch_size=self.config.batch_size, shuffle=False, pin_memory=self.cuda, sampler=None)
        self.dataloader_valid = DataLoader(GoogleSpeechCommandDataset(
            Path(environ["GSCPATH"]), mode="valid"),
            batch_size=self.config.batch_size, shuffle=False, pin_memory=self.cuda, sampler=None)

    def init(self):
        pass

    def train_one_epoch(self):
        self.model.train()
        for batch_id, (x, y) in enumerate(self.dataloader_train):
            if self.cuda:
                x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            if random.random() < 0.0:
                x = self.cutout(x)
                x = self.augment(x)
            output = self.model(x)
            loss = self.loss(output, y)
            loss.backward()
            self.optimizer.step()
            if batch_id % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch + 1,
                    batch_id * len(x),
                    len(self.dataloader_train.dataset),
                    100. * batch_id / len(self.dataloader_train),
                    loss.data.item() / self.config.batch_size))

    def validate(self, mode="test"):
        assert mode in ["test", "valid"]
        if mode == "test":
            loader = self.dataloader_test
        else:
            loader = self.dataloader_valid
        self.model.eval()
        test_loss = 0
        correct_predictions = 0
        with t.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                test_loss += self.loss(output, y).data.item()
                pred = output.max(1, keepdim=True)[1]
                correct_predictions += pred.eq(y.view_as(pred)).sum().item()
        print(f"Testing results after epoch {self.current_epoch + 1}:\n"
              f"Accuracy: {correct_predictions / len(loader.dataset)}\n"
              f"Loss: {test_loss / len(loader.dataset)}")
        return test_loss / len(loader.dataset), correct_predictions / len(loader.dataset)

    def finalize(self):
        pass
