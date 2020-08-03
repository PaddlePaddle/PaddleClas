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
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = [
    "RegNetX_200MF", "RegNetX_4GF", "RegNetX_32GF", "RegNetY_200MF",
    "RegNetY_4GF", "RegNetY_32GF"
]


class RegNet():
    def __init__(self, w_a, w_0, w_m, d, group_w, bot_mul, q=8, se_on=False):
        self.w_a = w_a
        self.w_0 = w_0
        self.w_m = w_m
        self.d = d
        self.q = q
        self.group_w = group_w
        self.bot_mul = bot_mul
        # Stem type
        self.stem_type = "simple_stem_in"
        # Stem width
        self.stem_w = 32
        # Block type
        self.block_type = "res_bottleneck_block"
        # Stride of each stage
        self.stride = 2
        # Squeeze-and-Excitation (RegNetY)
        self.se_on = se_on
        self.se_r = 0.25

    def quantize_float(self, f, q):
        """Converts a float to closest non-zero int divisible by q."""
        return int(round(f / q) * q)

    def adjust_ws_gs_comp(self, ws, bms, gs):
        """Adjusts the compatibility of widths and groups."""
        ws_bot = [int(w * b) for w, b in zip(ws, bms)]
        gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
        ws_bot = [
            self.quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)
        ]
        ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
        return ws, gs

    def get_stages_from_blocks(self, ws, rs):
        """Gets ws/ds of network at each stage from per block values."""
        ts = [
            w != wp or r != rp
            for w, wp, r, rp in zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
        ]
        s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
        s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
        return s_ws, s_ds

    def generate_regnet(self, w_a, w_0, w_m, d, q=8):
        """Generates per block ws from RegNet parameters."""
        assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
        ws_cont = np.arange(d) * w_a + w_0
        ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
        ws = w_0 * np.power(w_m, ks)
        ws = np.round(np.divide(ws, q)) * q
        num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
        ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
        return ws, num_stages, max_stage, ws_cont

    def init_weights(self, op_type, filter_size=0, num_channels=0, name=None):
        if op_type == 'conv':
            fan_out = num_channels * filter_size * filter_size
            param_attr = ParamAttr(
                name=name + "_weights",
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=math.sqrt(2.0 / fan_out)))
            bias_attr = False
        elif op_type == 'bn':
            param_attr = ParamAttr(
                name=name + "_scale",
                initializer=fluid.initializer.Constant(0.0))
            bias_attr = ParamAttr(
                name=name + "_offset",
                initializer=fluid.initializer.Constant(0.0))
        elif op_type == 'final_bn':
            param_attr = ParamAttr(
                name=name + "_scale",
                initializer=fluid.initializer.Constant(1.0))
            bias_attr = ParamAttr(
                name=name + "_offset",
                initializer=fluid.initializer.Constant(0.0))
        return param_attr, bias_attr

    def net(self, input, class_dim=1000):
        # Generate RegNet ws per block
        b_ws, num_s, max_s, ws_cont = self.generate_regnet(
            self.w_a, self.w_0, self.w_m, self.d, self.q)
        # Convert to per stage format
        ws, ds = self.get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [self.group_w for _ in range(num_s)]
        bms = [self.bot_mul for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = self.adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [self.stride for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = self.se_r

        # Construct the model
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        conv = self.conv_bn_layer(
            input=input,
            num_filters=self.stem_w,
            filter_size=3,
            stride=2,
            padding=1,
            act='relu',
            name="stem_conv")
        # Construct the stages
        for block, (d, w_out, stride, bm, gw) in enumerate(stage_params):
            for i in range(d):
                # Stride apply to the first block of the stage
                b_stride = stride if i == 0 else 1
                conv_name = 's' + str(block + 1) + '_b' + str(i +
                                                              1)  # chr(97 + i)
                conv = self.bottleneck_block(
                    input=conv,
                    num_filters=w_out,
                    stride=b_stride,
                    bm=bm,
                    gw=gw,
                    se_r=self.se_r,
                    name=conv_name)
        pool = fluid.layers.pool2d(
            input=conv, pool_type='avg', global_pooling=True)
        out = fluid.layers.fc(
            input=pool,
            size=class_dim,
            param_attr=ParamAttr(
                name="fc_0.w_0",
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=0.01)),
            bias_attr=ParamAttr(
                name="fc_0.b_0", initializer=fluid.initializer.Constant(0.0)))
        return out

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      padding=0,
                      act=None,
                      name=None,
                      final_bn=False):
        conv = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            act=None,
            name=name + '.conv2d.output.1')
        bn_name = name + '_bn'

        return fluid.layers.batch_norm(
            input=conv,
            act=act,
            name=bn_name + '.output.1',
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance', )

    def shortcut(self, input, ch_out, stride, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1:
            return self.conv_bn_layer(
                input=input,
                num_filters=ch_out,
                filter_size=1,
                stride=stride,
                padding=0,
                act=None,
                name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, bm, gw, se_r, name):
        # Compute the bottleneck width
        w_b = int(round(num_filters * bm))
        # Compute the number of groups
        num_gs = w_b // gw
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=w_b,
            filter_size=1,
            stride=1,
            padding=0,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=w_b,
            filter_size=3,
            stride=stride,
            padding=1,
            groups=num_gs,
            act='relu',
            name=name + "_branch2b")
        # Squeeze-and-Excitation (SE)
        if self.se_on:
            w_se = int(round(input.shape[1] * se_r))
            conv1 = self.squeeze_excitation(
                input=conv1,
                num_channels=w_b,
                reduction_channels=w_se,
                name=name + "_branch2se")

        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters,
            filter_size=1,
            stride=1,
            padding=0,
            act=None,
            name=name + "_branch2c",
            final_bn=True)

        short = self.shortcut(
            input, num_filters, stride, name=name + "_branch1")

        return fluid.layers.elementwise_add(x=short, y=conv2, act='relu')

    def squeeze_excitation(self,
                           input,
                           num_channels,
                           reduction_channels,
                           name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        fan_out = num_channels
        squeeze = fluid.layers.conv2d(
            input=pool,
            num_filters=reduction_channels,
            filter_size=1,
            act='relu',
            param_attr=ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=math.sqrt(2.0 / fan_out)),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        excitation = fluid.layers.conv2d(
            input=squeeze,
            num_filters=num_channels,
            filter_size=1,
            act='sigmoid',
            param_attr=ParamAttr(
                initializer=fluid.initializer.NormalInitializer(
                    loc=0.0, scale=math.sqrt(2.0 / fan_out)),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale


def RegNetX_200MF():
    model = RegNet(
        w_a=36.44, w_0=24, w_m=2.49, d=13, group_w=8, bot_mul=1.0, q=8)
    return model


def RegNetX_4GF():
    model = RegNet(
        w_a=38.65, w_0=96, w_m=2.43, d=23, group_w=40, bot_mul=1.0, q=8)
    return model


def RegNetX_32GF():
    model = RegNet(
        w_a=69.86, w_0=320, w_m=2.0, d=23, group_w=168, bot_mul=1.0, q=8)
    return model


def RegNetY_200MF():
    model = RegNet(
        w_a=36.44,
        w_0=24,
        w_m=2.49,
        d=13,
        group_w=8,
        bot_mul=1.0,
        q=8,
        se_on=True)
    return model


def RegNetY_4GF():
    model = RegNet(
        w_a=31.41,
        w_0=96,
        w_m=2.24,
        d=22,
        group_w=64,
        bot_mul=1.0,
        q=8,
        se_on=True)
    return model


def RegNetY_32GF():
    model = RegNet(
        w_a=115.89,
        w_0=232,
        w_m=2.53,
        d=20,
        group_w=232,
        bot_mul=1.0,
        q=8,
        se_on=True)
    return model
