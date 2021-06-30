from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class SelfONN1DLayer(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int
    q: int
    stride: int
    padding: int
    dilation: int
    groups: int
    padding_mode: str
    sampling_factor: int
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 q: int,
                 stride: int = 1,
                 padding: int = -1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',
                 sampling_factor: int = 1):
        super(SelfONN1DLayer, self).__init__()

        # Validity checks
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.q = q
        self.stride = stride
        if padding == -1:
            self.padding = int(np.ceil(self.kernel_size / 2)) - 1
        else:
            self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.padding_mode = padding_mode
        self.sampling_factor = sampling_factor

        self.weight = nn.Parameter(torch.Tensor(out_channels, q * in_channels // groups, kernel_size))  # C x (QxK) x D

        self.reset_parameters_like_torch()

    def reset_parameters(self):
        bound = 0.01
        nn.init.uniform_(self.bias, -bound, bound)
        with torch.no_grad():
            for q in range(self.q):
                # torch.nn.init.kaiming_uniform_(self.weights[q], a=math.sqrt(5))
                nn.init.xavier_uniform_(self.weight[q])
                # self.weights.data[q] /= factorial(q+1)

    def reset_parameters_like_torch(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.weight, gain=gain)
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward_slow(self, x):  # SEPARABLE FOR POOL OPERATION
        raise NotImplementedError

    def forward(self, x):
        # SelfONN stacking
        x = torch.cat([(x ** i) for i in range(1, self.q + 1)], dim=1)
        if self.padding_mode != 'zeros':
            # The pad argument is an iterable for both left and right sides in 1D
            x = F.pad(x, pad=(self.padding, self.padding), mode=self.padding_mode)
            x = F.conv1d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=0,
                         dilation=self.dilation,
                         groups=self.groups)
        else:
            x = F.conv1d(x,
                         weight=self.weight,
                         bias=self.bias,
                         stride=self.stride,
                         padding=self.padding,
                         dilation=self.dilation,
                         groups=self.groups)

        # Subsampling
        if self.sampling_factor > 1:
            x = torch.nn.functional.max_pool1d(x, kernel_size=self.sampling_factor, padding=0)
        elif self.sampling_factor < 1:
            x = torch.nn.functional.interpolate(x, scale_factor=abs(self.sampling_factor))

        return x

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}, q={q}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SelfONNDenseLayer(nn.Module):
    """
    Dense layer implementation of self-ONNs.
    """

    def __init__(self, in_features, out_features, q, sampling_factor=1, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.q = q
        self.sampling_factor = sampling_factor
        self.bias = bias
        self.layer = SelfONN1DLayer(in_channels=1,
                                    out_channels=out_features,
                                    kernel_size=in_features,
                                    pad=0, q=q, bias=bias)

    def forward(self, x):
        """
        Input x should be flattened, i.e. (batch_size x features).
        Works after Flatten(), similar to torch.nn.Linear.
        """

        # x is reshaped to be (batch_size x 1 x features) due to convolution.
        # the output, (batch_size x out_channels x 1), is reshaped to be (batch_size x out_channels).
        batch_size = x.shape[0]
        return self.layer(x.view(batch_size, 1, -1)).view(batch_size, -1)


class SelfONN1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, sampling_factor, idx=-1, dir=[], pad=-1, debug=False,
                 output=False, vis=None):
        super().__init__()
        self.q1 = SelfONN1DLayer(in_channels, out_channels, kernel_size, 1, sampling_factor=sampling_factor, pad=pad)
        self.q3 = SelfONN1DLayer(in_channels, out_channels, kernel_size, 3, sampling_factor=sampling_factor, pad=pad)
        self.q5 = SelfONN1DLayer(in_channels, out_channels, kernel_size, 5, sampling_factor=sampling_factor, pad=pad)
        self.q7 = SelfONN1DLayer(in_channels, out_channels, kernel_size, 7, sampling_factor=sampling_factor, pad=pad)

    def forward(self, x):
        # Input to layer
        out = torch.stack([self.q1(x), self.q3(x), self.q5(x), self.q7(x)], 0).sum(0)
        return out


if __name__ == "__main__":
    model = SelfONN1DLayer(64, 64, 11, 5, bias=False)
    # model = nn.Conv1d(
    #    64,
    #    64,
    #    11,
    #    padding=5,
    #    bias=False,
    #    groups=64,
    # )
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
