import torch.nn as nn
import torch.nn.functional as F

from utils.weights_initalizer import weights_init


class BasicNet(nn.Module):

    def __init__(self, output_features=20):
        super(BasicNet, self).__init__()
        self.nl = nn.PReLU()
        self.kernel_size = 5

        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=20,
                               kernel_size=self.kernel_size)
        self.conv2 = nn.Conv2d(in_channels=20,
                               out_channels=20,
                               kernel_size=self.kernel_size)
        self.dropout2d = nn.Dropout2d(0.3)
        self.dense1 = nn.Linear(in_features=200,
                                out_features=40)
        self.dropout = nn.Dropout(0.3)
        self.dense2 = nn.Linear(in_features=40,
                                out_features=output_features)

        self.apply(weights_init)

    def forward(self, x):
        x = self.nl(F.max_pool2d(self.conv1(x), 2))
        x = self.nl(F.max_pool2d(self.dropout2d(self.conv2(x)), 2))
        x = self.nl(self.dense1(x.view(x.size(0), -1)))
        x = self.dense2(self.dropout(x))
        return F.log_softmax(x, dim=1)

