from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.nn as nn
from paddle import Tensor
from ppcls.arch.utils import _calculate_fan_in_and_fan_out, kaiming_uniform_
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ["LeNetPlus"]


class Conv_PReLU_x2(nn.Layer):
    """basic block including 4 layer: conv->prelu->conv->prelu

    Args:
        in_channels (int): in_channels.
        out_channels (int): out_channels.
        kernel_size (int): kernel_size.
        stride (int): stride.
        padding (int): padding.

    Returns:
        nn.Layer: Conv_PReLU_x2 block.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int) -> nn.Layer:
        super().__init__()
        self.conv_1 = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        self.prelu_1 = nn.PReLU()
        self.conv_2 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding)
        self.prelu_2 = nn.PReLU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.prelu_1(self.conv_1(x))
        x = self.prelu_2(self.conv_2(x))
        return x


class LeNet(nn.Layer):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self):
        super(LeNet, self).__init__()
        self.block1 = Conv_PReLU_x2(1, 32, 5, 1, 2)
        self.block2 = Conv_PReLU_x2(32, 64, 5, 1, 2)
        self.block3 = Conv_PReLU_x2(64, 128, 5, 1, 2)

        self.max_pool2d = nn.MaxPool2D(kernel_size=2)

        def _init_func(m: nn.Layer):
            if isinstance(m, nn.Conv2D):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.initializer.Uniform(-bound, bound)(m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.initializer.Uniform(-bound, bound)(m.bias)

        self.apply(_init_func)

    def forward(self, x: Tensor) -> Tensor:
        x = self.block1(x)
        x = self.max_pool2d(x)

        x = self.block2(x)
        x = self.max_pool2d(x)

        x = self.block3(x)
        x = self.max_pool2d(x)

        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def LeNetPlus(pretrained=False, use_ssld=False, **kwargs):
    model = LeNet(**kwargs)
    _load_pretrained(pretrained, model, None, use_ssld=use_ssld)
    return model
