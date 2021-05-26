# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]

# VGG config
# key: VGG network depth
# value: conv num in different blocks
NET_CONFIG = {
    11: [1, 1, 2, 2, 2],
    13: [2, 2, 2, 2, 2],
    16: [2, 2, 3, 3, 3],
    19: [2, 2, 4, 4, 4]
}


def VGG11(**args):
    """
    VGG11
    Args:
        kwargs: 
            class_num: int=1000. Output dim of last fc layer.
            stop_grad_layers: int=0. The parameters in blocks which index larger than `stop_grad_layers`, will be set `param.trainable=False`
    Returns:
        model: nn.Layer. Specific `VGG11` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[11], **args)
    return model


def VGG13(**args):
    """
    VGG13
    Args:
        kwargs: 
            class_num: int=1000. Output dim of last fc layer.
            stop_grad_layers: int=0. The parameters in blocks which index larger than `stop_grad_layers`, will be set `param.trainable=False`
    Returns:
        model: nn.Layer. Specific `VGG11` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[13], **args)
    return model


def VGG16(**args):
    """
    VGG16
    Args:
        kwargs: 
            class_num: int=1000. Output dim of last fc layer.
            stop_grad_layers: int=0. The parameters in blocks which index larger than `stop_grad_layers`, will be set `param.trainable=False`
    Returns:
        model: nn.Layer. Specific `VGG11` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[16], **args)
    return model


def VGG19(**args):
    """
    VGG19
    Args:
        kwargs: 
            class_num: int=1000. Output dim of last fc layer.
            stop_grad_layers: int=0. The parameters in blocks which index larger than `stop_grad_layers`, will be set `param.trainable=False`
    Returns:
        model: nn.Layer. Specific `VGG11` model depends on args.
    """
    model = VGGNet(config=NET_CONFIG[19], **args)
    return model


class ConvBlock(TheseusLayer):
    def __init__(self, input_channels, output_channels, groups):
        super(ConvBlock, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False)

        self._pool = MaxPool2D(kernel_size=2, stride=2, padding=0)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
            x = F.relu(x)
        if self.groups == 3 or self.groups == 4:
            x = self._conv_3(x)
            x = F.relu(x)
        if self.groups == 4:
            x = self._conv_4(x)
            x = F.relu(x)
        x = self._pool(x)
        return x


class VGGNet(TheseusLayer):
    def __init__(self, config, stop_grad_layers=0, class_num=1000):
        super().__init__()

        self.stop_grad_layers = stop_grad_layers

        self._conv_block_1 = ConvBlock(3, 64, config[0])
        self._conv_block_2 = ConvBlock(64, 128, config[1])
        self._conv_block_3 = ConvBlock(128, 256, config[2])
        self._conv_block_4 = ConvBlock(256, 512, config[3])
        self._conv_block_5 = ConvBlock(512, 512, config[4])

        self._relu = nn.ReLU()

        for idx, block in enumerate([
                self._conv_block_1, self._conv_block_2, self._conv_block_3,
                self._conv_block_4, self._conv_block_5
        ]):
            if self.stop_grad_layers >= idx + 1:
                for param in block.parameters():
                    param.trainable = False

        self._drop = Dropout(p=0.5, mode="downscale_in_infer")
        self._fc1 = Linear(7 * 7 * 512, 4096)
        self._fc2 = Linear(4096, 4096)
        self._out = Linear(4096, class_num)

    def forward(self, inputs):
        x = self._conv_block_1(inputs)
        x = self._conv_block_2(x)
        x = self._conv_block_3(x)
        x = self._conv_block_4(x)
        x = self._conv_block_5(x)
        x = x.flatten(start_axis=1, stop_axis=-1)
        x = self._fc1(x)
        x = self._relu(x)
        x = self._drop(x)
        x = self._fc2(x)
        x = self._relu(x)
        x = self._drop(x)
        x = self._out(x)
        return x
