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
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout

import math

__all__ = ["SE_ResNeXt50_vd_32x4d", "SE_ResNeXt50_vd_32x4d", "SENet154_vd"]


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(
            self,
            num_channels,
            num_filters,
            filter_size,
            stride=1,
            groups=1,
            is_vd_mode=False,
            act=None,
            name=None, ):
        super(ConvBNLayer, self).__init__()

        self.is_vd_mode = is_vd_mode
        self._pool2d_avg = Pool2D(
            pool_size=2, pool_stride=2, pool_padding=0, pool_type='avg')
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)
        bn_name = name + '_bn'
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 cardinality,
                 reduction_ratio,
                 shortcut=True,
                 if_first=False,
                 name=None):
        super(BottleneckBlock, self).__init__()

        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name='conv' + name + '_x1')
        self.conv1 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters,
            filter_size=3,
            groups=cardinality,
            stride=stride,
            act='relu',
            name='conv' + name + '_x2')
        self.conv2 = ConvBNLayer(
            num_channels=num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            filter_size=1,
            act=None,
            name='conv' + name + '_x3')
        self.scale = SELayer(
            num_channels=num_filters * 2 if cardinality == 32 else num_filters,
            num_filters=num_filters * 2 if cardinality == 32 else num_filters,
            reduction_ratio=reduction_ratio,
            name='fc_' + name)

        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters * 2
                if cardinality == 32 else num_filters,
                filter_size=1,
                stride=1,
                is_vd_mode=False if if_first else True,
                name='conv' + name + '_prj')

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        conv2 = self.conv2(conv1)
        scale = self.scale(conv2)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)
        y = fluid.layers.elementwise_add(x=short, y=scale, act='relu')
        return y


class SELayer(fluid.dygraph.Layer):
    def __init__(self, num_channels, num_filters, reduction_ratio, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = Pool2D(pool_type='avg', global_pooling=True)

        self._num_channels = num_channels

        med_ch = int(num_channels / reduction_ratio)
        stdv = 1.0 / math.sqrt(num_channels * 1.0)
        self.squeeze = Linear(
            num_channels,
            med_ch,
            act="relu",
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + "_sqz_weights"),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))

        stdv = 1.0 / math.sqrt(med_ch * 1.0)
        self.excitation = Linear(
            med_ch,
            num_filters,
            act="sigmoid",
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + "_exc_weights"),
            bias_attr=ParamAttr(name=name + '_exc_offset'))

    def forward(self, input):
        pool = self.pool2d_gap(input)
        pool = fluid.layers.reshape(pool, shape=[-1, self._num_channels])
        squeeze = self.squeeze(pool)
        excitation = self.excitation(squeeze)
        excitation = fluid.layers.reshape(
            excitation, shape=[-1, self._num_channels, 1, 1])
        out = input * excitation
        return out


class ResNeXt(fluid.dygraph.Layer):
    def __init__(self, layers=50, class_dim=1000, cardinality=32):
        super(ResNeXt, self).__init__()

        self.layers = layers
        self.cardinality = cardinality
        self.reduction_ratio = 16
        supported_layers = [50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)
        supported_cardinality = [32, 64]
        assert cardinality in supported_cardinality, \
            "supported cardinality is {} but input cardinality is {}" \
            .format(supported_cardinality, cardinality)
        if layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_channels = [128, 256, 512, 1024]
        num_filters = [128, 256, 512,
                       1024] if cardinality == 32 else [256, 512, 1024, 2048]

        self.conv1_1 = ConvBNLayer(
            num_channels=3,
            num_filters=64,
            filter_size=3,
            stride=2,
            act='relu',
            name="conv1_1")
        self.conv1_2 = ConvBNLayer(
            num_channels=64,
            num_filters=64,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_2")
        self.conv1_3 = ConvBNLayer(
            num_channels=64,
            num_filters=128,
            filter_size=3,
            stride=1,
            act='relu',
            name="conv1_3")

        self.pool2d_max = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

        self.block_list = []
        n = 1 if layers == 50 or layers == 101 else 3
        for block in range(len(depth)):
            n += 1
            shortcut = False
            for i in range(depth[block]):
                bottleneck_block = self.add_sublayer(
                    'bb_%d_%d' % (block, i),
                    BottleneckBlock(
                        num_channels=num_channels[block] if i == 0 else
                        num_filters[block] * int(64 // self.cardinality),
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        cardinality=self.cardinality,
                        reduction_ratio=self.reduction_ratio,
                        shortcut=shortcut,
                        if_first=block == 0,
                        name=str(n) + '_' + str(i + 1)))
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = Pool2D(
            pool_size=7, pool_type='avg', global_pooling=True)

        self.pool2d_avg_channels = num_channels[-1] * 2

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_dim,
            param_attr=ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name="fc6_weights"),
            bias_attr=ParamAttr(name="fc6_offset"))

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        y = self.pool2d_max(y)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


def SE_ResNeXt50_vd_32x4d(**args):
    model = ResNeXt(layers=50, cardinality=32, **args)
    return model


def SE_ResNeXt101_vd_32x4d(**args):
    model = ResNeXt(layers=101, cardinality=32, **args)
    return model


def SENet154_vd(**args):
    model = ResNeXt(layers=152, cardinality=64, **args)
    return model
