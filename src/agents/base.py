import logging
import os

import torch as t
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler

t.backends.cudnn.benchmark = True


class BaseAgent:

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")
        self.model: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: _LRScheduler = None
        self.current_epoch = 0
        self.loss_history = []

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
        checkpoint = t.load(checkpoint_file)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.loss_history = checkpoint["loss_history"]
        if "scheduler_state_dict" in checkpoint:
            self.scheduler = checkpoint["scheduler_state_dict"]
            self.scheduler.optimizer = self.optimizer
            self.scheduler.step(self.current_epoch)
            self.scheduler.max_steps = self.config.epochs
        self.logger.info("Loaded model from {}".format(checkpoint_file))

    def save_checkpoint(self, checkpoint_file="checkpoint.pth.tar"):
        if self.scheduler is not None:
            t.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
                "scheduler_state_dict": self.scheduler
            }, checkpoint_file)
        else:
            t.save({
                "epoch": self.current_epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history
            }, checkpoint_file)

    def init(self):
        raise NotImplementedError

    def run(self):
        if self.config.checkpoint_file and os.path.exists(self.config.checkpoint_file):
            print("Attempting to load checkpoint from", self.config.checkpoint_file)
            try:
                self.load_checkpoint(self.config.checkpoint_file)
            except Exception as e:
                self.logger.warning("Failed to load checkpoint file!")
                print(e)
        try:
            if self.config.mode == "train":
                self.train()
            # loss, accuracy = self.validate(mode="valid")
        except KeyboardInterrupt:
            self.logger.info("Interrupt received, shutting down gracefully.")

    def train(self):
        print(self.model)
        for _ in tqdm(range(self.current_epoch, self.config.epochs), desc="Training progress"):
            for g in self.optimizer.param_groups:
                print("Learning rate for next epoch:", g["lr"])
            print(self.config.checkpoint_dir)
            self.train_one_epoch()
            loss, accuracy = self.validate(mode="valid")
            self.current_epoch += 1
            if self.scheduler is not None:
                self.scheduler.step(self.current_epoch)
            self.loss_history.append((loss, accuracy))
            # lowest_loss_tuple = sorted(self.loss_history, key=lambda x: x[0], reverse=False)[0]
            lowest_loss_tuple = sorted(self.loss_history, key=lambda x: x[1], reverse=True)[0]
            if accuracy == lowest_loss_tuple[1] or self.current_epoch % 10 == 0:
                self.save_checkpoint(
                    os.path.join(
                        self.config.checkpoint_dir,
                        self.model.module._get_name() + str(self.current_epoch) + ".pth"))
            for i, entry in enumerate(self.loss_history):
                print(f"Epoch {i + 1}:", entry[0], entry[1])
            print("Highest accuracy: ")
            print(f"Loss: {lowest_loss_tuple[0]}, accuracy: {lowest_loss_tuple[1]}")
        best_checkpoint_id = self.loss_history.index(lowest_loss_tuple) + 1
        self.load_checkpoint(
            os.path.join(
                self.config.checkpoint_dir,
                self.model.module._get_name() + str(self.current_epoch) + ".pth"))
        loss, accuracy = self.validate(mode="test", print_results=False)
        print("Testing results: ")
        print(f"Loss: {loss}, accuracy: {accuracy}")


def train_one_epoch(self):
    raise NotImplementedError


def validate(self, mode):
    raise NotImplementedError


def finalize(self):
    raise NotImplementedError
