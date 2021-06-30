import torch as t
from torch.nn.functional import softmax, avg_pool2d, relu, log_softmax
from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, Linear, Module, Sequential, ReLU, Tanh, Dropout, Sigmoid
from torchsummary import summary
import torch

torch.autograd.set_detect_anomaly(True)
from fastonn import SelfONNLayer

from utils.weights_initalizer import weights_init

n_classes = 20
Q = 1


class EdgeSpeechNet(Module):

    def __init__(self, target_n):
        super().__init__()
        self.encoder = None
        self._make_encoder()
        self.decoder = Linear(in_features=45, out_features=target_n)
        # self.apply(weights_init)

    def _make_encoder(self):
        layers = []
        for entry in self.__class__.architecture:
            layer, kwargs = entry
            layers.append(layer(**kwargs))
        self.encoder = Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = avg_pool2d(x, (x.size(2), x.size(3)))
        x = x.squeeze(2).squeeze(2)
        x = self.decoder(x)
        return log_softmax(x, dim=1)


class ESNConv2d(Module):

    def __init__(self, in_channels, out_channels, padding_mode="zeros"):
        super().__init__()
        self.conv = Conv2d(in_channels=in_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           bias=False,
                           padding=1,
                           padding_mode=padding_mode)
        self.act = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class ESNResBlock(Module):

    def __init__(self, in_out_channels, mid_channels, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.prebn = BatchNorm2d(in_out_channels)
        self.conv1 = ESNConv2d(in_channels=in_out_channels, out_channels=mid_channels)
        self.bn = BatchNorm2d(mid_channels)
        self.conv2 = ESNConv2d(in_channels=mid_channels, out_channels=in_out_channels)

    def forward(self, x):
        residual_input = x
        if not self.first:
            x = self.prebn(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        x += residual_input
        return x


class Scale(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        x = x - torch.max(x) / 2
        factor = torch.max(torch.max(x), torch.abs(torch.min(x)))
        x = torch.div(x, factor)
        return x



class EdgeSpeechNetA(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 39}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 20, "first": True}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 15}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 22}),
        (ESNResBlock, {"in_out_channels": 39, "mid_channels": 25}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 45}),
        #(Conv2d, {"in_channels": 45, "out_channels": 45, "kernel_size": 1}),
        #(Tanh, {}),
        #(SelfONNLayer, {"in_channels": 45, "out_channels": 45, "q": 5, "kernel_size": 3, "padding": 1})
    ]


class EdgeSpeechNetB(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 30}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8, "first": True}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 10}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 8}),
        (ESNResBlock, {"in_out_channels": 30, "mid_channels": 11}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 45}),
    ]


class EdgeSpeechNetC(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 24}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6, "first": True}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 9}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 12}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 5}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 6}),
        (ESNResBlock, {"in_out_channels": 24, "mid_channels": 2}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 45}),
    ]


class EdgeSpeechNetD(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 45}),
        (AvgPool2d, {"kernel_size": 2}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 30, "first": True}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 33}),
        (ESNResBlock, {"in_out_channels": 45, "mid_channels": 35}),
    ]


import torch.nn.functional as F
from torch.nn import Dropout2d

kernel_size = 3


class TestNet(Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv2d(1, 20, kernel_size=kernel_size, padding=1, stride=1,
                            dilation=1)
        self.conv2 = Conv2d(20, 20, kernel_size=kernel_size, padding=1, stride=1,
                            dilation=1)
        self.conv2_drop = Dropout2d()
        # input size of fc1: np.floor( (np.floor((H-4)/2) - 4)/2 ) * np.floor( (np.floor((W-4)/2) - 4)/2 ) * 20
        self.fc1 = Linear(800, 40)
        self.fc2 = Linear(40, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
