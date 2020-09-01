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
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.initializer import MSRA
import math

__all__ = [
    "ShuffleNetV2_x0_25", "ShuffleNetV2_x0_33", "ShuffleNetV2_x0_5",
    "ShuffleNetV2_x1_0", "ShuffleNetV2_x1_5", "ShuffleNetV2_x2_0",
    "ShuffleNetV2_swish"
]


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.shape[0], x.shape[1], x.shape[
        2], x.shape[3]
    channels_per_group = num_channels // groups

    # reshape
    x = fluid.layers.reshape(
        x=x, shape=[batchsize, groups, channels_per_group, height, width])

    x = fluid.layers.transpose(x=x, perm=[0, 2, 1, 3, 4])
    # flatten
    x = fluid.layers.reshape(
        x=x, shape=[batchsize, num_channels, height, width])
    return x


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 channels=None,
                 num_groups=1,
                 if_act=True,
                 act='relu',
                 name=None,
                 use_cudnn=True):
        super(ConvBNLayer, self).__init__()
        self._if_act = if_act
        assert act in ['relu', 'swish'], \
            "supported act are {} but your act is {}".format(
                ['relu', 'swish'], act)
        self._act = act
        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            act=None,
            use_cudnn=use_cudnn,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + "_weights"),
            bias_attr=False)

        self._batch_norm = BatchNorm(
            num_filters,
            param_attr=ParamAttr(name=name + "_bn_scale"),
            bias_attr=ParamAttr(name=name + "_bn_offset"),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")

    def forward(self, inputs, if_act=True):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        if self._if_act:
            y = fluid.layers.relu(
                y) if self._act == 'relu' else fluid.layers.swish(y)
        return y


class InvertedResidualUnit(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 benchmodel,
                 act='relu',
                 name=None):
        super(InvertedResidualUnit, self).__init__()
        assert stride in [1, 2], \
            "supported stride are {} but your stride is {}".format([
                                                                   1, 2], stride)
        self.benchmodel = benchmodel
        oup_inc = num_filters // 2
        inp = num_channels
        if benchmodel == 1:
            self._conv_pw = ConvBNLayer(
                num_channels=num_channels // 2,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv1')
            self._conv_dw = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                act=act,
                use_cudnn=False,
                name='stage_' + name + '_conv2')
            self._conv_linear = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv3')
        else:
            # branch1
            self._conv_dw_1 = ConvBNLayer(
                num_channels=num_channels,
                num_filters=inp,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=inp,
                if_act=False,
                act=act,
                use_cudnn=False,
                name='stage_' + name + '_conv4')
            self._conv_linear_1 = ConvBNLayer(
                num_channels=inp,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv5')
            # branch2
            self._conv_pw_2 = ConvBNLayer(
                num_channels=num_channels,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv1')
            self._conv_dw_2 = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=3,
                stride=stride,
                padding=1,
                num_groups=oup_inc,
                if_act=False,
                act=act,
                use_cudnn=False,
                name='stage_' + name + '_conv2')
            self._conv_linear_2 = ConvBNLayer(
                num_channels=oup_inc,
                num_filters=oup_inc,
                filter_size=1,
                stride=1,
                padding=0,
                num_groups=1,
                if_act=True,
                act=act,
                name='stage_' + name + '_conv3')

    def forward(self, inputs):
        if self.benchmodel == 1:
            x1, x2 = fluid.layers.split(
                inputs,
                num_or_sections=[inputs.shape[1] // 2, inputs.shape[1] // 2],
                dim=1)
            x2 = self._conv_pw(x2)
            x2 = self._conv_dw(x2)
            x2 = self._conv_linear(x2)
            out = fluid.layers.concat([x1, x2], axis=1)
        else:
            x1 = self._conv_dw_1(inputs)
            x1 = self._conv_linear_1(x1)

            x2 = self._conv_pw_2(inputs)
            x2 = self._conv_dw_2(x2)
            x2 = self._conv_linear_2(x2)
            out = fluid.layers.concat([x1, x2], axis=1)

        return channel_shuffle(out, 2)


class ShuffleNet(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000, scale=1.0, act='relu'):
        super(ShuffleNet, self).__init__()
        self.scale = scale
        self.class_dim = class_dim
        stage_repeats = [4, 8, 4]

        if scale == 0.25:
            stage_out_channels = [-1, 24, 24, 48, 96, 512]
        elif scale == 0.33:
            stage_out_channels = [-1, 24, 32, 64, 128, 512]
        elif scale == 0.5:
            stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif scale == 1.0:
            stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif scale == 1.5:
            stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif scale == 2.0:
            stage_out_channels = [-1, 24, 224, 488, 976, 2048]
        else:
            raise NotImplementedError("This scale size:[" + str(scale) +
                                      "] is not implemented!")
        # 1. conv1
        self._conv1 = ConvBNLayer(
            num_channels=3,
            num_filters=stage_out_channels[1],
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            act=act,
            name='stage1_conv')
        self._max_pool = Pool2D(
            pool_type='max', pool_size=3, pool_stride=2, pool_padding=1)

        # 2. bottleneck sequences
        self._block_list = []
        i = 1
        in_c = int(32 * scale)
        for idxstage in range(len(stage_repeats)):
            numrepeat = stage_repeats[idxstage]
            output_channel = stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    block = self.add_sublayer(
                        str(idxstage + 2) + '_' + str(i + 1),
                        InvertedResidualUnit(
                            num_channels=stage_out_channels[idxstage + 1],
                            num_filters=output_channel,
                            stride=2,
                            benchmodel=2,
                            act=act,
                            name=str(idxstage + 2) + '_' + str(i + 1)))
                    self._block_list.append(block)
                else:
                    block = self.add_sublayer(
                        str(idxstage + 2) + '_' + str(i + 1),
                        InvertedResidualUnit(
                            num_channels=output_channel,
                            num_filters=output_channel,
                            stride=1,
                            benchmodel=1,
                            act=act,
                            name=str(idxstage + 2) + '_' + str(i + 1)))
                    self._block_list.append(block)

        # 3. last_conv
        self._last_conv = ConvBNLayer(
            num_channels=stage_out_channels[-2],
            num_filters=stage_out_channels[-1],
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act,
            name='conv5')

        # 4. pool
        self._pool2d_avg = Pool2D(pool_type='avg', global_pooling=True)
        self._out_c = stage_out_channels[-1]
        # 5. fc
        self._fc = Linear(
            stage_out_channels[-1],
            class_dim,
            param_attr=ParamAttr(name='fc6_weights'),
            bias_attr=ParamAttr(name='fc6_offset'))

    def forward(self, inputs):
        y = self._conv1(inputs)
        y = self._max_pool(y)
        for inv in self._block_list:
            y = inv(y)
        y = self._last_conv(y)
        y = self._pool2d_avg(y)
        y = fluid.layers.reshape(y, shape=[-1, self._out_c])
        y = self._fc(y)
        return y


def ShuffleNetV2_x0_25(**args):
    model = ShuffleNetV2(scale=0.25, **args)
    return model


def ShuffleNetV2_x0_33(**args):
    model = ShuffleNet(scale=0.33, **args)
    return model


def ShuffleNetV2_x0_5(**args):
    model = ShuffleNet(scale=0.5, **args)
    return model


def ShuffleNetV2(**args):
    model = ShuffleNet(scale=1.0, **args)
    return model


def ShuffleNetV2_x1_5(**args):
    model = ShuffleNet(scale=1.5, **args)
    return model


def ShuffleNetV2_x2_0(**args):
    model = ShuffleNet(scale=2.0, **args)
    return model


def ShuffleNetV2_swish(**args):
    model = ShuffleNet(scale=1.0, act='swish', **args)
    return model
