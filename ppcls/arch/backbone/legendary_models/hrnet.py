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

import math
import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer

__all__ = [
    "HRNet_W18_C",
    "HRNet_W30_C",
    "HRNet_W32_C",
    "HRNet_W40_C",
    "HRNet_W44_C",
    "HRNet_W48_C",
    "HRNet_W60_C",
    "HRNet_W64_C",
    "SE_HRNet_W18_C",
    "SE_HRNet_W30_C",
    "SE_HRNet_W32_C",
    "SE_HRNet_W40_C",
    "SE_HRNet_W44_C",
    "SE_HRNet_W48_C",
    "SE_HRNet_W60_C",
    "SE_HRNet_W64_C",
]


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act="relu"):
        super(ConvBNLayer, self).__init__()

        self._conv = nn.Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            bias_attr=False)
        self._batch_norm = nn.BatchNorm(
            num_filters,
            act=act)

    def forward(self, x, res_dict=None):
        y = self._conv(x)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se,
                 stride=1,
                 downsample=False):
        super(BottleneckBlock, self).__init__()

        self.has_se = has_se
        self.downsample = downsample

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act="relu")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="relu")
        self.conv3 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None)

        if self.downsample:
            self.conv_down = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 4,
                filter_size=1,
                act=None)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters * 4,
                num_filters=num_filters * 4,
                reduction_ratio=16)

    def forward(self, x, res_dict=None):
        residual = x
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)

        if self.downsample:
            residual = self.conv_down(x)

        if self.has_se:
            conv3 = self.se(conv3)

        y = paddle.add(x=residual, y=conv3)
        y = F.relu(y)
        return y


class BasicBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 has_se=False):
        super(BasicBlock, self).__init__()

        self.has_se = has_se

        self.conv1 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act="relu")
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            stride=1,
            act=None)

        if self.has_se:
            self.se = SELayer(
                num_channels=num_filters,
                num_filters=num_filters,
                reduction_ratio=16)

    def forward(self, input):
        residual = input
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)

        if self.has_se:
            conv2 = self.se(conv2)

        y = paddle.add(x=residual, y=conv2)
        y = F.relu(y)
        return y


class SELayer(TheseusLayer):
    def __init__(self, num_channels, num_filters, reduction_ratio):
        super(SELayer, self).__init__()

        self.pool2d_gap = AdaptiveAvgPool2D(1)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = nn.Linear(
            num_channels,
            med_ch,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv)))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = nn.Linear(
            med_ch,
            num_filters,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv)))

    def forward(self, input, res_dict=None):
        pool = self.pool2d_gap(input)
        pool = paddle.squeeze(pool, axis=[2, 3])
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        excitation = paddle.unsqueeze(excitation, axis=[2, 3])
        out = input * excitation
        return out


class Stage(TheseusLayer):
    def __init__(self,
                 num_modules,
                 num_filters,
                 has_se=False):
        super(Stage, self).__init__()

        self._num_modules = num_modules

        self.stage_func_list = nn.LayerList()
        for i in range(num_modules):
            self.stage_func_list.append(
                HighResolutionModule(
                    num_filters=num_filters,
                    has_se=has_se))

    def forward(self, input, res_dict=None):
        out = input
        for idx in range(self._num_modules):
            out = self.stage_func_list[idx](out)
        return out


class HighResolutionModule(TheseusLayer):
    def __init__(self,
                 num_filters,
                 has_se=False):
        super(HighResolutionModule, self).__init__()

        self.basic_block_list = nn.LayerList()

        for i in range(len(num_filters)):
            self.basic_block_list.append(
                nn.Sequential(*[
                    BasicBlock(
                        num_channels=num_filters[i],
                        num_filters=num_filters[i],
                        has_se=has_se) for j in range(4)]))

        self.fuse_func = FuseLayers(
            in_channels=num_filters,
            out_channels=num_filters)

    def forward(self, input, res_dict=None):
        outs = []
        for idx, input in enumerate(input):
            conv = input
            basic_block_list = self.basic_block_list[idx]
            for basic_block_func in basic_block_list:
                conv = basic_block_func(conv)
            outs.append(conv)
        out = self.fuse_func(outs)
        return out


class FuseLayers(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels):
        super(FuseLayers, self).__init__()

        self._actual_ch = len(in_channels)
        self._in_channels = in_channels

        self.residual_func_list = nn.LayerList()
        for i in range(len(in_channels)):
            for j in range(len(in_channels)):
                if j > i:
                    self.residual_func_list.append(
                        ConvBNLayer(
                            num_channels=in_channels[j],
                            num_filters=out_channels[i],
                            filter_size=1,
                            stride=1,
                            act=None))
                elif j < i:
                    pre_num_filters = in_channels[j]
                    for k in range(i - j):
                        if k == i - j - 1:
                            self.residual_func_list.append(
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[i],
                                    filter_size=3,
                                    stride=2,
                                    act=None))
                            pre_num_filters = out_channels[i]
                        else:
                            self.residual_func_list.append(
                                ConvBNLayer(
                                    num_channels=pre_num_filters,
                                    num_filters=out_channels[j],
                                    filter_size=3,
                                    stride=2,
                                    act="relu"))
                            pre_num_filters = out_channels[j]

    def forward(self, input, res_dict=None):
        outs = []
        residual_func_idx = 0
        for i in range(len(self._in_channels)):
            residual = input[i]
            for j in range(len(self._in_channels)):
                if j > i:
                    y = self.residual_func_list[residual_func_idx](input[j])
                    residual_func_idx += 1

                    y = F.upsample(y, scale_factor=2**(j - i), mode="nearest")
                    residual = paddle.add(x=residual, y=y)
                elif j < i:
                    y = input[j]
                    for k in range(i - j):
                        y = self.residual_func_list[residual_func_idx](y)
                        residual_func_idx += 1

                    residual = paddle.add(x=residual, y=y)

            residual = F.relu(residual)
            outs.append(residual)

        return outs


class LastClsOut(TheseusLayer):
    def __init__(self,
                 num_channel_list,
                 has_se,
                 num_filters_list=[32, 64, 128, 256]):
        super(LastClsOut, self).__init__()

        self.func_list = nn.LayerList()
        for idx in range(len(num_channel_list)):
            self.func_list.append(
                BottleneckBlock(
                    num_channels=num_channel_list[idx],
                    num_filters=num_filters_list[idx],
                    has_se=has_se,
                    downsample=True))

    def forward(self, inputs, res_dict=None):
        outs = []
        for idx, input in enumerate(inputs):
            out = self.func_list[idx](input)
            outs.append(out)
        return outs


class HRNet(TheseusLayer):
    def __init__(self, width=18, has_se=False, class_dim=1000):
        super(HRNet, self).__init__()

        self.width = width
        self.has_se = has_se
        self._class_dim = class_dim

        channels_2 = [self.width, self.width * 2]
        channels_3 = [self.width, self.width * 2, self.width * 4]
        channels_4 = [self.width, self.width * 2, self.width * 4, self.width * 8]

        self.conv_layer1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu')

        self.conv_layer1_2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu')

        self.layer1 = nn.Sequential(*[
            BottleneckBlock(
                num_channels=64 if i == 0 else 256,
                num_filters=64,
                has_se=has_se,
                stride=1,
                downsample=True if i == 0 else False)
            for i in range(4)
        ])

        self.tr1_1 = ConvBNLayer(
            num_channels=256,
            num_filters=width,
            filter_size=3)
        self.tr1_2 = ConvBNLayer(
            num_channels=256,
            num_filters=width * 2,
            filter_size=3,
            stride=2
        )

        self.st2 = Stage(
            num_modules=1,
            num_filters=channels_2,
            has_se=self.has_se)

        self.tr2 = ConvBNLayer(
            num_channels=width * 2,
            num_filters=width * 4,
            filter_size=3,
            stride=2
        )
        self.st3 = Stage(
            num_modules=4,
            num_filters=channels_3,
            has_se=self.has_se)

        self.tr3 = ConvBNLayer(
            num_channels=width * 4,
            num_filters=width * 8,
            filter_size=3,
            stride=2
        )

        self.st4 = Stage(
            num_modules=3,
            num_filters=channels_4,
            has_se=self.has_se)

        # classification
        num_filters_list = [32, 64, 128, 256]
        self.last_cls = LastClsOut(
            num_channel_list=channels_4,
            has_se=self.has_se,
            num_filters_list=num_filters_list)

        last_num_filters = [256, 512, 1024]
        self.cls_head_conv_list = nn.LayerList()
        for idx in range(3):
            self.cls_head_conv_list.append(
                    ConvBNLayer(
                        num_channels=num_filters_list[idx] * 4,
                        num_filters=last_num_filters[idx],
                        filter_size=3,
                        stride=2))

        self.conv_last = ConvBNLayer(
            num_channels=1024,
            num_filters=2048,
            filter_size=1,
            stride=1)

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        stdv = 1.0 / math.sqrt(2048 * 1.0)

        self.out = nn.Linear(
            2048,
            class_dim,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv)))

    def forward(self, input, res_dict=None):
        conv1 = self.conv_layer1_1(input)
        conv2 = self.conv_layer1_2(conv1)

        la1 = self.layer1(conv2)

        tr1_1 = self.tr1_1(la1)
        tr1_2 = self.tr1_2(la1)
        st2 = self.st2([tr1_1, tr1_2])

        tr2 = self.tr2(st2[-1])
        st2.append(tr2)
        st3 = self.st3(st2)

        tr3 = self.tr3(st3[-1])
        st3.append(tr3)
        st4 = self.st4(st3)

        last_cls = self.last_cls(st4)

        y = last_cls[0]
        for idx in range(3):
            y = paddle.add(last_cls[idx + 1], self.cls_head_conv_list[idx](y))

        y = self.conv_last(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, y.shape[1]])
        y = self.out(y)
        return y


def HRNet_W18_C(**args):
    model = HRNet(width=18, **args)
    return model


def HRNet_W30_C(**args):
    model = HRNet(width=30, **args)
    return model


def HRNet_W32_C(**args):
    model = HRNet(width=32, **args)
    return model


def HRNet_W40_C(**args):
    model = HRNet(width=40, **args)
    return model


def HRNet_W44_C(**args):
    model = HRNet(width=44, **args)
    return model


def HRNet_W48_C(**args):
    model = HRNet(width=48, **args)
    return model


def HRNet_W60_C(**args):
    model = HRNet(width=60, **args)
    return model


def HRNet_W64_C(**args):
    model = HRNet(width=64, **args)
    return model


def SE_HRNet_W18_C(**args):
    model = HRNet(width=18, has_se=True, **args)
    return model


def SE_HRNet_W30_C(**args):
    model = HRNet(width=30, has_se=True, **args)
    return model


def SE_HRNet_W32_C(**args):
    model = HRNet(width=32, has_se=True, **args)
    return model


def SE_HRNet_W40_C(**args):
    model = HRNet(width=40, has_se=True, **args)
    return model


def SE_HRNet_W44_C(**args):
    model = HRNet(width=44, has_se=True, **args)
    return model


def SE_HRNet_W48_C(**args):
    model = HRNet(width=48, has_se=True, **args)
    return model


def SE_HRNet_W60_C(**args):
    model = HRNet(width=60, has_se=True, **args)
    return model


def SE_HRNet_W64_C(**args):
    model = HRNet(width=64, has_se=True, **args)
    return model
