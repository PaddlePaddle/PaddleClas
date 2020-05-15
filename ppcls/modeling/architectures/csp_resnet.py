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

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "CSPResNet50_leaky", "CSPResNet50_mish", "CSPResNet101_leaky",
    "CSPResNet101_mish"
]


class CSPResNet():
    def __init__(self, layers=50, act="leaky_relu"):
        self.layers = layers
        self.act = act

    def net(self, input, class_dim=1000, data_format="NCHW"):
        layers = self.layers
        supported_layers = [50, 101]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(
                supported_layers, layers)

        if layers == 50:
            depth = [3, 3, 5, 2]
        elif layers == 101:
            depth = [3, 3, 22, 2]

        num_filters = [64, 128, 256, 512]

        conv = self.conv_bn_layer(
            input=input,
            num_filters=64,
            filter_size=7,
            stride=2,
            act=self.act,
            name="conv1",
            data_format=data_format)
        conv = fluid.layers.pool2d(
            input=conv,
            pool_size=2,
            pool_stride=2,
            pool_padding=0,
            pool_type='max',
            data_format=data_format)

        for block in range(len(depth)):
            conv_name = "res" + str(block + 2) + chr(97)
            if block != 0:
                conv = self.conv_bn_layer(
                    input=conv,
                    num_filters=num_filters[block],
                    filter_size=3,
                    stride=2,
                    act=self.act,
                    name=conv_name + "_downsample",
                    data_format=data_format)

            # split
            left = conv
            right = conv
            if block == 0:
                ch = num_filters[block]
            else:
                ch = num_filters[block] * 2
            right = self.conv_bn_layer(
                input=right,
                num_filters=ch,
                filter_size=1,
                act=self.act,
                name=conv_name + "_right_first_route",
                data_format=data_format)

            for i in range(depth[block]):
                conv_name = "res" + str(block + 2) + chr(97 + i)

                right = self.bottleneck_block(
                    input=right,
                    num_filters=num_filters[block],
                    stride=1,
                    name=conv_name,
                    data_format=data_format)

            # route
            left = self.conv_bn_layer(
                input=left,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act=self.act,
                name=conv_name + "_left_route",
                data_format=data_format)
            right = self.conv_bn_layer(
                input=right,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                act=self.act,
                name=conv_name + "_right_route",
                data_format=data_format)
            conv = fluid.layers.concat([left, right], axis=1)

            conv = self.conv_bn_layer(
                input=conv,
                num_filters=num_filters[block] * 2,
                filter_size=1,
                stride=1,
                act=self.act,
                name=conv_name + "_merged_transition",
                data_format=data_format)

        pool = fluid.layers.pool2d(
            input=conv,
            pool_type='avg',
            global_pooling=True,
            data_format=data_format)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=fluid.param_attr.ParamAttr(
                name="fc_0.w_0",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_0.b_0"))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None,
                      data_format='NCHW'):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False,
            name=name + '.conv2d.output.1',
            data_format=data_format)

        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]
        bn = fluid.layers.batch_norm(
            input=conv,
            act=None,
            name=bn_name + '.output.1',
            param_attr=ParamAttr(name=bn_name + '_scale'),
            bias_attr=ParamAttr(bn_name + '_offset'),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance',
            data_layout=data_format)
        if act == "relu":
            bn = fluid.layers.relu(bn)
        elif act == "leaky_relu":
            bn = fluid.layers.leaky_relu(bn)
        elif act == "mish":
            bn = self._mish(bn)
        return bn

    def _mish(self, input):
        return input * fluid.layers.tanh(self._softplus(input))

    def _softplus(self, input):
        expf = fluid.layers.exp(fluid.layers.clip(input, -200, 50))
        return fluid.layers.log(1 + expf)

    def shortcut(self, input, ch_out, stride, is_first, name, data_format):
        if data_format == 'NCHW':
            ch_in = input.shape[1]
        else:
            ch_in = input.shape[-1]
        if ch_in != ch_out or stride != 1 or is_first is True:
            return self.conv_bn_layer(
                input, ch_out, 1, stride, name=name, data_format=data_format)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name, data_format):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act="leaky_relu",
            name=name + "_branch2a",
            data_format=data_format)
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act="leaky_relu",
            name=name + "_branch2b",
            data_format=data_format)
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 2,
            filter_size=1,
            act=None,
            name=name + "_branch2c",
            data_format=data_format)

        short = self.shortcut(
            input,
            num_filters * 2,
            stride,
            is_first=False,
            name=name + "_branch1",
            data_format=data_format)

        ret = short + conv2
        ret = fluid.layers.leaky_relu(ret, alpha=0.1)
        return ret


def CSPResNet50_leaky():
    model = CSPResNet(layers=50, act="leaky_relu")
    return model


def CSPResNet50_mish():
    model = CSPResNet(layers=50, act="mish")
    return model


def CSPResNet101_leaky():
    model = CSPResNet(layers=101, act="leaky_relu")
    return model


def CSPResNet101_mish():
    model = CSPResNet(layers=101, act="mish")
    return model
