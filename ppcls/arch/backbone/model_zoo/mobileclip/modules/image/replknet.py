# -*- coding: utf-8 -*-
from typing import Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from mobileclip.modules.common.mobileone import MobileOneBlock


class ReparamLargeKernelConv(nn.Layer):
    """Building Block of RepLKNet"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: nn.Layer = nn.GELU(),
    ) -> None:
        super(ReparamLargeKernelConv, self).__init__()

        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = kernel_size // 2

        # Check if SE is requested
        if use_se:
            # Simple SE implementation for Paddle
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Conv2D(out_channels, int(out_channels * 0.25), 1),
                nn.ReLU(),
                nn.Conv2D(int(out_channels * 0.25), out_channels, 1),
                nn.Sigmoid()
            )
        else:
            self.se = nn.Identity()

        if inference_mode:
            self.lkb_reparam = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=self.padding,
                dilation=1,
                groups=groups,
                bias_attr=True,
            )
        else:
            self.lkb_origin = self._conv_bn(
                kernel_size=kernel_size, padding=self.padding
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(
                    kernel_size=small_kernel, padding=small_kernel // 2
                )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Apply forward pass."""
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(x)
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out += self.small_conv(x)

        if isinstance(self.se, nn.Sequential):
            out = out * self.se(out)
        else:
            out = self.se(out)
            
        return self.activation(out)

    def _conv_bn(self, kernel_size: int, padding: int = 0) -> nn.Sequential:
        return nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias_attr=False,
            )),
            ('bn', nn.BatchNorm2D(num_features=self.out_channels))
        )
