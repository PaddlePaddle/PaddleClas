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
import paddle.nn.functional as F
from paddle.nn import Conv2d, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2d, MaxPool2d, AvgPool2d
from paddle.regularizer import L2Decay

import math

__all__ = [
    "MobileNetV3_small_x0_35", "MobileNetV3_small_x0_5",
    "MobileNetV3_small_x0_75", "MobileNetV3_small_x1_0",
    "MobileNetV3_small_x1_25", "MobileNetV3_large_x0_35",
    "MobileNetV3_large_x0_5", "MobileNetV3_large_x0_75",
    "MobileNetV3_large_x1_0", "MobileNetV3_large_x1_25"
]


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MobileNetV3(nn.Layer):
    def __init__(self, scale=1.0, model_name="small", class_dim=1000):
        super(MobileNetV3, self).__init__()

        inplanes = 16
        if model_name == "large":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, False, "relu", 1],
                [3, 64, 24, False, "relu", 2],
                [3, 72, 24, False, "relu", 1],
                [5, 72, 40, True, "relu", 2],
                [5, 120, 40, True, "relu", 1],
                [5, 120, 40, True, "relu", 1],
                [3, 240, 80, False, "hard_swish", 2],
                [3, 200, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 184, 80, False, "hard_swish", 1],
                [3, 480, 112, True, "hard_swish", 1],
                [3, 672, 112, True, "hard_swish", 1],
                [5, 672, 160, True, "hard_swish", 2],
                [5, 960, 160, True, "hard_swish", 1],
                [5, 960, 160, True, "hard_swish", 1],
            ]
            self.cls_ch_squeeze = 960
            self.cls_ch_expand = 1280
        elif model_name == "small":
            self.cfg = [
                # k, exp, c,  se,     nl,  s,
                [3, 16, 16, True, "relu", 2],
                [3, 72, 24, False, "relu", 2],
                [3, 88, 24, False, "relu", 1],
                [5, 96, 40, True, "hard_swish", 2],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 240, 40, True, "hard_swish", 1],
                [5, 120, 48, True, "hard_swish", 1],
                [5, 144, 48, True, "hard_swish", 1],
                [5, 288, 96, True, "hard_swish", 2],
                [5, 576, 96, True, "hard_swish", 1],
                [5, 576, 96, True, "hard_swish", 1],
            ]
            self.cls_ch_squeeze = 576
            self.cls_ch_expand = 1280
        else:
            raise NotImplementedError(
                "mode[{}_model] is not implemented!".format(model_name))

        self.conv1 = ConvBNLayer(
            in_c=3,
            out_c=make_divisible(inplanes * scale),
            filter_size=3,
            stride=2,
            padding=1,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv1")

        self.block_list = []
        i = 0
        inplanes = make_divisible(inplanes * scale)
        for (k, exp, c, se, nl, s) in self.cfg:
            self.block_list.append(
                ResidualUnit(
                    in_c=inplanes,
                    mid_c=make_divisible(scale * exp),
                    out_c=make_divisible(scale * c),
                    filter_size=k,
                    stride=s,
                    use_se=se,
                    act=nl,
                    name="conv" + str(i + 2)))
            self.add_sublayer(
                sublayer=self.block_list[-1], name="conv" + str(i + 2))
            inplanes = make_divisible(scale * c)
            i += 1

        self.last_second_conv = ConvBNLayer(
            in_c=inplanes,
            out_c=make_divisible(scale * self.cls_ch_squeeze),
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act="hard_swish",
            name="conv_last")

        self.pool = AdaptiveAvgPool2d(1)

        self.last_conv = Conv2d(
            in_channels=make_divisible(scale * self.cls_ch_squeeze),
            out_channels=self.cls_ch_expand,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name="last_1x1_conv_weights"),
            bias_attr=False)

        self.out = Linear(
            self.cls_ch_expand,
            class_dim,
            weight_attr=ParamAttr("fc_weights"),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs, label=None, dropout_prob=0.2):
        x = self.conv1(inputs)
        for block in self.block_list:
            x = block(x)
        x = self.last_second_conv(x)
        x = self.pool(x)
        x = self.last_conv(x)
        x = F.hard_swish(x)
        x = F.dropout(x=x, p=dropout_prob)
        x = paddle.reshape(x, shape=[x.shape[0], x.shape[1]])
        x = self.out(x)

        return x


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 if_act=True,
                 act=None,
                 use_cudnn=True,
                 name=""):
        super(ConvBNLayer, self).__init__()
        self.if_act = if_act
        self.act = act
        self.conv = Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(
                name=name + "_bn_scale", regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(
                name=name + "_bn_offset", regularizer=L2Decay(0.0)),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            if self.act == "relu":
                x = F.relu(x)
            elif self.act == "hard_swish":
                x = F.hard_swish(x)
            else:
                print("The activation function is selected incorrectly.")
                exit()
        return x


class ResidualUnit(nn.Layer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 use_se,
                 act=None,
                 name=''):
        super(ResidualUnit, self).__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.if_se = use_se

        self.expand_conv = ConvBNLayer(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name=name + "_expand")
        self.bottleneck_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            num_groups=mid_c,
            if_act=True,
            act=act,
            name=name + "_depthwise")
        if self.if_se:
            self.mid_se = SEModule(mid_c, name=name + "_se")
        self.linear_conv = ConvBNLayer(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None,
            name=name + "_linear")

    def forward(self, inputs):
        x = self.expand_conv(inputs)
        x = self.bottleneck_conv(x)
        if self.if_se:
            x = self.mid_se(x)
        x = self.linear_conv(x)
        if self.if_shortcut:
            x = paddle.elementwise_add(inputs, x)
        return x


class SEModule(nn.Layer):
    def __init__(self, channel, reduction=4, name=""):
        super(SEModule, self).__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.conv1 = Conv2d(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name=name + "_1_weights"),
            bias_attr=ParamAttr(name=name + "_1_offset"))
        self.conv2 = Conv2d(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(name + "_2_weights"),
            bias_attr=ParamAttr(name=name + "_2_offset"))

    def forward(self, inputs):
        outputs = self.avg_pool(inputs)
        outputs = self.conv1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = F.hard_sigmoid(outputs)
        return paddle.multiply(x=inputs, y=outputs, axis=0)


def MobileNetV3_small_x0_35(**args):
    model = MobileNetV3(model_name="small", scale=0.35, **args)
    return model


def MobileNetV3_small_x0_5(**args):
    model = MobileNetV3(model_name="small", scale=0.5, **args)
    return model


def MobileNetV3_small_x0_75(**args):
    model = MobileNetV3(model_name="small", scale=0.75, **args)
    return model


def MobileNetV3_small_x1_0(**args):
    model = MobileNetV3(model_name="small", scale=1.0, **args)
    return model


def MobileNetV3_small_x1_25(**args):
    model = MobileNetV3(model_name="small", scale=1.25, **args)
    return model


def MobileNetV3_large_x0_35(**args):
    model = MobileNetV3(model_name="large", scale=0.35, **args)
    return model


def MobileNetV3_large_x0_5(**args):
    model = MobileNetV3(model_name="large", scale=0.5, **args)
    return model


def MobileNetV3_large_x0_75(**args):
    model = MobileNetV3(model_name="large", scale=0.75, **args)
    return model


def MobileNetV3_large_x1_0(**args):
    model = MobileNetV3(model_name="large", scale=1.0, **args)
    return model


def MobileNetV3_large_x1_25(**args):
    model = MobileNetV3(model_name="large", scale=1.25, **args)
    return model
