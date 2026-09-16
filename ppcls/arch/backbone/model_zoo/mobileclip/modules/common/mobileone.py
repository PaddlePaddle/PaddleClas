# -*- coding: utf-8 -*-
from typing import Union, Tuple

import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["MobileOneBlock", "reparameterize_model"]


class SEBlock(nn.Layer):
    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.expand = nn.Conv2D(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        b, c, h, w = inputs.shape
        x = F.adaptive_avg_pool2d(inputs, output_size=[1, 1])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = F.sigmoid(x)
        return inputs * x


class MobileOneBlock(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: nn.Layer = nn.GELU(),
    ) -> None:
        super(MobileOneBlock, self).__init__()
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        if use_se:
            self.se = SEBlock(out_channels)
        else:
            self.se = nn.Identity()

        if use_act:
            self.activation = activation
        else:
            self.activation = nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=True,
            )
        else:
            self.rbr_skip = (
                nn.BatchNorm2D(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )

            if num_conv_branches > 0:
                rbr_conv = list()
                for _ in range(self.num_conv_branches):
                    rbr_conv.append(
                        self._conv_bn(kernel_size=kernel_size, padding=padding)
                    )
                self.rbr_conv = nn.LayerList(rbr_conv)
            else:
                self.rbr_conv = None

            self.rbr_scale = None
            if not isinstance(kernel_size, int):
                kernel_size = kernel_size[0]
            if (kernel_size > 1) and use_scale_branch:
                self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def _conv_bn(self, kernel_size, padding):
        res = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                dilation=self.dilation,
                groups=self.groups,
                bias_attr=False,
            )),
            ('bn', nn.BatchNorm2D(num_features=self.out_channels))
        )
        return res

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.inference_mode:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        if self.rbr_conv is not None:
            for ix in range(self.num_conv_branches):
                out += self.rbr_conv[ix](x)

        return self.activation(self.se(out))


def reparameterize_model(model: nn.Layer) -> nn.Layer:
    model_copy = copy.deepcopy(model)
    for module in model_copy.sublayers():
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model_copy
