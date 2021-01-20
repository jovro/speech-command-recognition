import torch.nn as nn


class NegativeLogLikelihood(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.NLLLoss()

    def forward(self, prediction, target):
        return self.loss(prediction, target)
