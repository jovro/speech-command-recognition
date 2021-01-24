import logging

import torch as t
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class BaseAgent:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.current_epoch = 0

        self.cuda_available = t.cuda.is_available()
        if self.cuda_available and not self.config.cuda:
            self.logger.warning("WARNING: You have a CUDA device yet it is not enabled!")

        self.cuda = self.cuda_available and self.config.cuda

        if self.cuda:
            self.device = t.device("cuda")
            self.logger.info("Program will run on GPU/CUDA")
        else:
            self.device = t.device("cpu")
            self.logger.info("Program will run on CPU")

        if "random_seed" in self.config:
            t.manual_seed(config.random_seed)
            if self.cuda:
                t.cuda.manual_seed(config.random_seed)

    def load_checkpoint(self, checkpoint_file):
        assert self.model is not None
        assert self.optimizer is not None
        checkpoint = t.load(checkpoint_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]

    def save_checkpoint(self, checkpoint_file="checkpoint.pth.tar"):
        t.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }, checkpoint_file)

    def init(self):
        raise NotImplementedError

    def run(self):
        try:
            if self.config.mode == "train":
                self.train()
        except KeyboardInterrupt:
            self.logger.info("Interrupt received, shutting down gracefully.")

    def train(self):
        for _ in tqdm(range(1, self.config.epochs + 1), desc="Training progress"):
            self.train_one_epoch()
            self.validate(mode="test")
            self.current_epoch += 1

    def train_one_epoch(self):
        raise NotImplementedError

    def validate(self, mode):
        raise NotImplementedError

    def finalize(self):
        raise NotImplementedError
