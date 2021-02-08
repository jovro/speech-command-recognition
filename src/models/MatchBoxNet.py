from typing import List

import torch as t
import torch.nn as nn
import torch.nn.functional as f


class MatchBoxBlock(nn.Module):
    __constants__ = ["residual", "separable", "convs", "residual_layers"]

    def __init__(
            self,
            in_channels,
            out_channels,
            subblocks=3,
            kernel_size=11,
            stride=1,
            dilation=1,
            dropout=0.2,
            residual=True,
            separable=False,
    ):
        super().__init__()
        assert not (stride > 1 and dilation > 1), "Stride and dilation cannot be both higher than 1"

        padding_amount = (dilation * kernel_size) // 2 - 1 if dilation > 1 else kernel_size // 2

        self.separable = separable

        subblock_list = nn.ModuleList()

        subblock_list.extend(
            self.get_conv_and_batchnorm(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_amount,
                separable=separable
            )
        )

        for _ in range(subblocks - 2):
            subblock_list.extend(
                self.get_conv_and_batchnorm(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding_amount,
                    separable=separable
                )
            )

            subblock_list.extend(nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout)))

        subblock_list.extend(
            self.get_conv_and_batchnorm(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding_amount,
                separable=separable
            )
        )

        self.convs = subblock_list

        self.residual_layers = None
        if residual:
            res_list = nn.ModuleList()

            res_panes = [in_channels]
            for ip in res_panes:
                res = nn.ModuleList(
                    self.get_conv_and_batchnorm(
                        ip,
                        out_channels,
                        kernel_size=1
                    ))

                res_list.append(res)

            self.residual_layers = res_list

        self.mout = nn.Sequential(nn.ReLU(), nn.Dropout(p=dropout))

    def get_conv_and_batchnorm(
            self,
            in_channels,
            out_channels,
            kernel_size=11,
            stride=1,
            dilation=1,
            padding=0,
            bias=False,
            separable=False
    ):
        if separable:
            layers = [
                nn.Conv1d(
                    in_channels,
                    in_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                    groups=in_channels,
                ),
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    dilation=1,
                    padding=0,
                    bias=bias,
                ),
            ]
        else:
            layers = [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    dilation=dilation,
                    padding=padding,
                    bias=bias,
                )
            ]

        layers.append(nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1))

        return layers

    def forward(self, input_: List[t.Tensor]) -> List[t.Tensor]:
        out = input_[-1]
        for layer in self.convs:
            out = layer(out)

        if self.residual_layers is not None:
            for i, layer in enumerate(self.residual_layers):
                res_out = input_[i]
                for res_layer in layer:
                    res_out = res_layer(res_out)
                out = out + res_out

        out = self.mout(out)
        return [out]


class MatchBoxNet(nn.Module):
    # https://arxiv.org/abs/2004.08531
    def __init__(self, num_classes, b=3, r=1, c=64):
        super().__init__()
        self.prologue = MatchBoxBlock(64, 128, kernel_size=11, subblocks=1, separable=True, residual=False)

        self.residuals = self._get_block_layers(b=b, r=r, c=c, in_channels=128, initial_kernel_size=13)

        self.epilogue = nn.Sequential(
            MatchBoxBlock(64, 128, kernel_size=29, dilation=2, subblocks=1, separable=True,
                          residual=False),
            MatchBoxBlock(128, 128, kernel_size=1, subblocks=1, separable=False,
                          residual=False))
        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)
        self.output = nn.Linear(in_features=128, out_features=num_classes, bias=True)

    def _get_block_layers(self, b, r, c, in_channels, initial_kernel_size=13):
        assert b > 0 and r > 0 and c > 0
        layers = []
        # First layer might have different inchannels
        layers.append(
            MatchBoxBlock(
                in_channels=in_channels,
                out_channels=c,
                kernel_size=initial_kernel_size,
                subblocks=r,
                separable=True,
                residual=True
            )
        )
        kernel_size = initial_kernel_size + 2

        for i in range(0, b - 1):
            layers.append(
                MatchBoxBlock(
                    in_channels=c,
                    out_channels=c,
                    kernel_size=kernel_size,
                    subblocks=r,
                    separable=True,
                    residual=True
                )
            )
            kernel_size += 2

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prologue([x])
        x = self.residuals(x)
        x = self.epilogue(x)
        batch, in_channels, timesteps = x[0].size()
        x = self.pooling(x[0]).view(batch, in_channels)
        x = self.output(x)
        return f.log_softmax(x, dim=-1)
