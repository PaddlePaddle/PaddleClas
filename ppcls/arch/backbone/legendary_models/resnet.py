# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer



NET_CONFIG = {
    "18": {
        "block_type": "BasicBlock", "block_depth": [2, 2, 2, 2], "num_channels": [64, 64, 128, 256]},
    "34": {
        "block_type": "BasicBlock", "block_depth": [3, 4, 6, 3], "num_channels": [64, 64, 128, 256]},
    "50": {
        "block_type": "BottleneckBlock", "block_depth": [3, 4, 6, 3], "num_channels": [64, 256, 512, 1024]},
    "101": {
        "block_type": "BottleneckBlock", "block_depth": [3, 4, 23, 3], "num_channels": [64, 256, 512, 1024]},
    "152": {
        "block_type": "BottleneckBlock", "block_depth": [3, 8, 36, 3], "num_channels": [64, 256, 512, 1024]},
    "200": {
        "block_type": "BottleneckBlock", "block_depth": [3, 12, 48, 3], "num_channels": [64, 256, 512, 1024]},
}



class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 is_vd_mode=False,
                 act=None,
                 lr_mult=1.0):
        super(ConvBNLayer, self).__init__()
        self.is_vd_mode = is_vd_mode
        self.act = act
        self.avgpool = AvgPool2D(
            kernel_size=2, stride=2, padding=0, ceil_mode=True)
        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=False)
        self.bn = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.relu = nn.ReLU()

    def forward(self, x):
        if self.is_vd_mode:
            x = self.avgpool(x)
        x = self.conv(x)
        x = self.bn(x)
        if self.act:
            x = self.relu(x)
        return x


class BottleneckBlock(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0,
                ):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            lr_mult=lr_mult)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            lr_mult=lr_mult)
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            lr_mult=lr_mult)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                stride=stride if if_first else 1,
                is_vd_mode=False if if_first else True,
                lr_mult=lr_mult)
        self.relu = nn.ReLU()
        self.shortcut = shortcut

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)
        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class BasicBlock(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 shortcut=True,
                 if_first=False,
                 lr_mult=1.0):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            lr_mult=lr_mult)
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            lr_mult=lr_mult)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride if if_first else 1,
                is_vd_mode=False if if_first else True,
                lr_mult=lr_mult)

        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv0(x)
        x = self.conv1(x)

        if self.shortcut:
            short = identity
        else:
            short = self.short(identity)

        x = paddle.add(x=x, y=short)
        x = self.relu(x)
        return x


class ResNet(TheseusLayer):
    """ResNet model from
    `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Parameters
    ----------
    config : dict of string and list
        Information of whole model.
    version : str, "vb" and "vd"
        Different version of ResNet, version vd can perform better.
    class_dim : int, default 1000
        Number of classification classes.
    lr_mult_list : list of float
        Control the learning rate of different stages
    """
    def __init__(self,
                 config,
                 version="vd",
                 class_dim=1000,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(ResNet, self).__init__()

        self.cfg = config
        self.lr_mult_list = lr_mult_list
        self.is_vd_mode = version == "vd"
        
        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(
            self.lr_mult_list
        ) == 5, "lr_mult_list length should be 5 but got {}".format(
            len(self.lr_mult_list))

        self.num_filters = [64, 128, 256, 512]
        self.channels_mult = 1 if self.cfg["num_channels"][-1] == 256 else 4
        self.stem_cfg = {
            "vb": [[3, 64, 7, 2]],
            "vd": [[3, 32, 3, 2],
                   [32, 32, 3, 1],
                   [32, 64, 3, 1]]}
        
        self.stem = nn.Sequential(*[
            ConvBNLayer(
                    num_channels=in_c,
                    num_filters=out_c,
                    filter_size=k,
                    stride=s,
                    act='relu',
                    lr_mult=self.lr_mult_list[0])
            for in_c, out_c, k, s in self.stem_cfg[version]
        ])
        
        self.maxpool = MaxPool2D(kernel_size=3, stride=2, padding=1)

        self.block_list = []
        for block in range(len(self.cfg["block_depth"])):
            shortcut = False
            for i in range(self.cfg["block_depth"][block]):
                self.block_list.append(
                    globals()[self.cfg["block_type"]](
                    num_channels=self.cfg["num_channels"][block]
                    if i == 0 else self.num_filters[block] * self.channels_mult,
                    num_filters=self.num_filters[block],
                    stride=2 if i == 0 and block != 0 else 1,
                    shortcut=shortcut,
                    if_first=block == i == 0 if version == "vd" else True,
                    lr_mult=self.lr_mult_list[block + 1]))
                shortcut = True
                
        self.blocks = nn.Sequential(*self.block_list)

        self.avgpool = AdaptiveAvgPool2D(1)

        self.avgpool_channels = self.cfg["num_channels"][-1] * 2

        stdv = 1.0 / math.sqrt(self.avgpool_channels * 1.0)

        self.out = Linear(
            self.avgpool_channels,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv)))

    def forward(self, x):
        x = self.stem(x)
        x = self.maxpool(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = paddle.reshape(x, shape=[-1, self.avgpool_channels])
        x = self.out(x)
        return x


def ResNet18(**args):
    model = ResNet(config=NET_CONFIG["18"], version="vb", **args)
    return model

def ResNet18_vd(**args):
    model = ResNet(config=NET_CONFIG["18"], version="vd", **args)
    return model

def ResNet50(**args):
    model = ResNet(config=NET_CONFIG["50"], version="vb", **args)
    return model

def ResNet50_vd(**args):
    model = ResNet(config=NET_CONFIG["50"], version="vd", **args)
    return model

def ResNet101(**args):
    model = ResNet(config=NET_CONFIG["101"], version="vb", **args)
    return model

def ResNet101_vd(**args):
    model = ResNet(config=NET_CONFIG["101"], version="vd", **args)
    return model

def ResNet152(**args):
    model = ResNet(config=NET_CONFIG["152"], version="vb", **args)
    return model

def ResNet152_vd(**args):
    model = ResNet(config=NET_CONFIG["152"], version="vd", **args)
    return model

def ResNet200(**args):
    model = ResNet(config=NET_CONFIG["200"], version="vb", **args)
    return model

def ResNet200_vd(**args):
    model = ResNet(config=NET_CONFIG["200"], version="vd", **args)
    return model
    
