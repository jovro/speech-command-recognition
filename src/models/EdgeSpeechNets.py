import torch as t
from torch.nn.functional import log_softmax, avg_pool2d
from torch.nn import AvgPool2d, BatchNorm2d, Conv2d, Linear, Module, Sequential, ReLU, Tanh, Dropout
from torchsummary import summary

from utils.weights_initalizer import weights_init

n_classes = 20


class EdgeSpeechNet(Module):

    def __init__(self):
        super().__init__()
        self.encoder = None
        self._make_encoder()
        self.decoder = Linear(in_features=45, out_features=n_classes)
        self.apply(weights_init)

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
                           padding=(1, 1),
                           padding_mode=padding_mode)
        self.bn = BatchNorm2d(out_channels, affine=False)
        self.do = Tanh()

    def forward(self, x):
        return self.do(self.bn(self.conv(x)))


class EdgeSpeechNetA(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 20}),
        (ESNConv2d, {"in_channels": 20, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 15}),
        (ESNConv2d, {"in_channels": 15, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 25}),
        (ESNConv2d, {"in_channels": 25, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 22}),
        (ESNConv2d, {"in_channels": 22, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 22}),
        (ESNConv2d, {"in_channels": 22, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 25}),
        (ESNConv2d, {"in_channels": 25, "out_channels": 39}),
        (ESNConv2d, {"in_channels": 39, "out_channels": 45}),
    ]


class EdgeSpeechNetB(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 8}),
        (ESNConv2d, {"in_channels": 8, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 9}),
        (ESNConv2d, {"in_channels": 9, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 11}),
        (ESNConv2d, {"in_channels": 11, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 10}),
        (ESNConv2d, {"in_channels": 10, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 8}),
        (ESNConv2d, {"in_channels": 8, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 11}),
        (ESNConv2d, {"in_channels": 11, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 45}),
    ]


class EdgeSpeechNetC(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 6}),
        (ESNConv2d, {"in_channels": 6, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 9}),
        (ESNConv2d, {"in_channels": 9, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 12}),
        (ESNConv2d, {"in_channels": 12, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 6}),
        (ESNConv2d, {"in_channels": 6, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 5}),
        (ESNConv2d, {"in_channels": 5, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 6}),
        (ESNConv2d, {"in_channels": 6, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 2}),
        (ESNConv2d, {"in_channels": 2, "out_channels": 24}),
        (ESNConv2d, {"in_channels": 24, "out_channels": 45}),
    ]


class EdgeSpeechNetD(EdgeSpeechNet):
    architecture = [
        (ESNConv2d, {"in_channels": 1, "out_channels": 45}),
        (AvgPool2d, {"kernel_size": 2}),
        (ESNConv2d, {"in_channels": 45, "out_channels": 30}),
        (ESNConv2d, {"in_channels": 30, "out_channels": 45}),
        (ESNConv2d, {"in_channels": 45, "out_channels": 33}),
        (ESNConv2d, {"in_channels": 33, "out_channels": 45}),
        (ESNConv2d, {"in_channels": 45, "out_channels": 35}),
        (ESNConv2d, {"in_channels": 35, "out_channels": 45}),
    ]
