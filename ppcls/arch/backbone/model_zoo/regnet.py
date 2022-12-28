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

# Code was based on https://github.com/facebookresearch/pycls
# reference: https://arxiv.org/abs/1905.13214

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform
import math

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "RegNetX_200MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_200MF_pretrained.pdparams",
    "RegNetX_400MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_400MF_pretrained.pdparams",
    "RegNetX_600MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_600MF_pretrained.pdparams",
    "RegNetX_800MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_800MF_pretrained.pdparams",
    "RegNetX_1600MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_1600MF_pretrained.pdparams",
    "RegNetX_3200MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_3200MF_pretrained.pdparams",
    "RegNetX_4GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_4GF_pretrained.pdparams",
    "RegNetX_6400MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_6400MF_pretrained.pdparams",
    "RegNetX_8GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_8GF_pretrained.pdparams",
    "RegNetX_12GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_12GF_pretrained.pdparams",
    "RegNetX_16GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_16GF_pretrained.pdparams",
    "RegNetX_32GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetX_32GF_pretrained.pdparams",
    "RegNetY_200MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_200MF_pretrained.pdparams",
    "RegNetY_400MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_400MF_pretrained.pdparams",
    "RegNetY_600MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_600MF_pretrained.pdparams",
    "RegNetY_800MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_800MF_pretrained.pdparams",
    "RegNetY_1600MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_1600MF_pretrained.pdparams",
    "RegNetY_3200MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_3200MF_pretrained.pdparams",
    "RegNetY_4GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_4GF_pretrained.pdparams",
    "RegNetY_6400MF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_6400MF_pretrained.pdparams",
    "RegNetY_8GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_8GF_pretrained.pdparams",
    "RegNetY_12GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_12GF_pretrained.pdparams",
    "RegNetY_16GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_16GF_pretrained.pdparams",
    "RegNetY_32GF":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/RegNetY_32GF_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def adjust_ws_gs_comp(ws, bms, gs):
    """Adjusts the compatibility of widths and groups."""
    ws_bot = [int(w * b) for w, b in zip(ws, bms)]
    gs = [min(g, w_bot) for g, w_bot in zip(gs, ws_bot)]
    ws_bot = [quantize_float(w_bot, g) for w_bot, g in zip(ws_bot, gs)]
    ws = [int(w_bot / b) for w_bot, b in zip(ws_bot, bms)]
    return ws, gs


def get_stages_from_blocks(ws, rs):
    """Gets ws/ds of network at each stage from per block values."""
    ts = [
        w != wp or r != rp
        for w, wp, r, rp in zip(ws + [0], [0] + ws, rs + [0], [0] + rs)
    ]
    s_ws = [w for w, t in zip(ws, ts[:-1]) if t]
    s_ds = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()
    return s_ws, s_ds


def generate_regnet(w_a, w_0, w_m, d, q=8):
    """Generates per block ws from RegNet parameters."""
    assert w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % q == 0
    ws_cont = np.arange(d) * w_a + w_0
    ks = np.round(np.log(ws_cont / w_0) / np.log(w_m))
    ws = w_0 * np.power(w_m, ks)
    ws = np.round(np.divide(ws, q)) * q
    num_stages, max_stage = len(np.unique(ws)), ks.max() + 1
    ws, ws_cont = ws.astype(int).tolist(), ws_cont.tolist()
    return ws, num_stages, max_stage, ws_cont


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 padding=0,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(name=name + ".conv2d.output.1.w_0"),
            bias_attr=False)
        bn_name = name + "_bn"
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + ".output.1.w_0"),
            bias_attr=ParamAttr(bn_name + ".output.1.b_0"),
            moving_mean_name=bn_name + "_mean",
            moving_variance_name=bn_name + "_variance")

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class BottleneckBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 stride,
                 bm,
                 gw,
                 se_on,
                 se_r,
                 shortcut=True,
                 name=None):
        super(BottleneckBlock, self).__init__()

        w_b = int(round(num_filters * bm))
        w_se = int(round(num_channels * se_r))
        # Compute the number of groups
        num_gs = w_b // gw
        self.se_on = se_on
        if not shortcut:
            self.short = ConvBNLayer(
                num_channels=num_channels,
                num_filters=num_filters,
                filter_size=1,
                stride=stride,
                name=name + "_branch1")
        self.conv0 = ConvBNLayer(
            num_channels=num_channels,
            num_filters=w_b,
            filter_size=1,
            padding=0,
            act="relu",
            name=name + "_branch2a")
        self.conv1 = ConvBNLayer(
            num_channels=w_b,
            num_filters=w_b,
            filter_size=3,
            stride=stride,
            padding=1,
            groups=num_gs,
            act="relu",
            name=name + "_branch2b")
        if se_on:
            w_se = int(round(num_channels * se_r))
            self.se_block = SELayer(
                num_channels=w_b, num_filters=w_se, name=name + "_branch2se")
        self.conv2 = ConvBNLayer(
            num_channels=w_b,
            num_filters=num_filters,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        self.shortcut = shortcut

    def forward(self, inputs):
        y = self.conv0(inputs)
        conv1 = self.conv1(y)
        if self.se_on:
            conv1 = self.se_block(conv1)
        conv2 = self.conv2(conv1)

        if self.shortcut:
            short = inputs
        else:
            short = self.short(inputs)

        y = paddle.add(x=short, y=conv2)
        y = F.relu(y)
        return y


class SELayer(nn.Layer):
    def __init__(self, num_channels, num_filters, name=None):
        super(SELayer, self).__init__()

        self.pool2d_gap = AdaptiveAvgPool2D(1)
        self._num_channels = num_channels
        self.squeeze = Conv2D(num_channels, num_filters, 1, bias_attr=True)
        self.excitation = Conv2D(num_filters, num_channels, 1, bias_attr=True)

    def forward(self, input):
        pool = self.pool2d_gap(input)
        squeeze = self.squeeze(pool)
        squeeze = F.relu(squeeze)
        excitation = self.excitation(squeeze)
        excitation = F.sigmoid(excitation)
        out = input * excitation
        return out


class RegNet(nn.Layer):
    def __init__(self,
                 w_a,
                 w_0,
                 w_m,
                 d,
                 group_w,
                 bot_mul,
                 q=8,
                 se_on=False,
                 class_num=1000):
        super(RegNet, self).__init__()

        # Generate RegNet ws per block
        b_ws, num_s, max_s, ws_cont = generate_regnet(w_a, w_0, w_m, d, q)
        # Convert to per stage format
        ws, ds = get_stages_from_blocks(b_ws, b_ws)
        # Generate group widths and bot muls
        gws = [group_w for _ in range(num_s)]
        bms = [bot_mul for _ in range(num_s)]
        # Adjust the compatibility of ws and gws
        ws, gws = adjust_ws_gs_comp(ws, bms, gws)
        # Use the same stride for each stage
        ss = [2 for _ in range(num_s)]
        # Use SE for RegNetY
        se_r = 0.25
        # Construct the model
        # Group params by stage
        stage_params = list(zip(ds, ws, ss, bms, gws))
        # Construct the stem
        stem_type = "simple_stem_in"
        stem_w = 32
        block_type = "res_bottleneck_block"

        self.conv = ConvBNLayer(
            num_channels=3,
            num_filters=stem_w,
            filter_size=3,
            stride=2,
            padding=1,
            act="relu",
            name="stem_conv")

        self.block_list = []
        for block, (d, w_out, stride, bm, gw) in enumerate(stage_params):
            shortcut = False
            for i in range(d):
                num_channels = stem_w if block == i == 0 else in_channels
                # Stride apply to the first block of the stage
                b_stride = stride if i == 0 else 1
                conv_name = "s" + str(block + 1) + "_b" + str(i +
                                                              1)  # chr(97 + i)
                bottleneck_block = self.add_sublayer(
                    conv_name,
                    BottleneckBlock(
                        num_channels=num_channels,
                        num_filters=w_out,
                        stride=b_stride,
                        bm=bm,
                        gw=gw,
                        se_on=se_on,
                        se_r=se_r,
                        shortcut=shortcut,
                        name=conv_name))
                in_channels = w_out
                self.block_list.append(bottleneck_block)
                shortcut = True

        self.pool2d_avg = AdaptiveAvgPool2D(1)

        self.pool2d_avg_channels = w_out

        stdv = 1.0 / math.sqrt(self.pool2d_avg_channels * 1.0)

        self.out = Linear(
            self.pool2d_avg_channels,
            class_num,
            weight_attr=ParamAttr(
                initializer=Uniform(-stdv, stdv), name="fc_0.w_0"),
            bias_attr=ParamAttr(name="fc_0.b_0"))

    def forward(self, inputs):
        y = self.conv(inputs)
        for block in self.block_list:
            y = block(y)
        y = self.pool2d_avg(y)
        y = paddle.reshape(y, shape=[-1, self.pool2d_avg_channels])
        y = self.out(y)
        return y


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


def RegNetX_200MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=36.44,
        w_0=24,
        w_m=2.49,
        d=13,
        group_w=8,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_200MF"], use_ssld=use_ssld)
    return model


def RegNetX_400MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=24.48,
        w_0=24,
        w_m=2.54,
        d=22,
        group_w=16,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_400MF"], use_ssld=use_ssld)
    return model


def RegNetX_600MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=36.97,
        w_0=48,
        w_m=2.24,
        d=16,
        group_w=24,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_600MF"], use_ssld=use_ssld)
    return model


def RegNetX_800MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=35.73,
        w_0=56,
        w_m=2.28,
        d=16,
        group_w=16,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_800MF"], use_ssld=use_ssld)
    return model


def RegNetX_1600MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=34.01,
        w_0=80,
        w_m=2.25,
        d=18,
        group_w=24,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_1600MF"], use_ssld=use_ssld)
    return model


def RegNetX_3200MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=26.31,
        w_0=88,
        w_m=2.25,
        d=25,
        group_w=48,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_3200MF"], use_ssld=use_ssld)
    return model


def RegNetX_4GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=38.65,
        w_0=96,
        w_m=2.43,
        d=23,
        group_w=40,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_4GF"], use_ssld=use_ssld)
    return model


def RegNetX_6400MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=60.83,
        w_0=184,
        w_m=2.07,
        d=17,
        group_w=56,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_6400MF"], use_ssld=use_ssld)
    return model


def RegNetX_8GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=49.56,
        w_0=80,
        w_m=2.88,
        d=23,
        group_w=120,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_8GF"], use_ssld=use_ssld)
    return model


def RegNetX_12GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=73.36,
        w_0=168,
        w_m=2.37,
        d=19,
        group_w=112,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_12GF"], use_ssld=use_ssld)
    return model


def RegNetX_16GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=55.59,
        w_0=216,
        w_m=2.1,
        d=22,
        group_w=128,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_16GF"], use_ssld=use_ssld)
    return model


def RegNetX_32GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=69.86,
        w_0=320,
        w_m=2.0,
        d=23,
        group_w=168,
        bot_mul=1.0,
        q=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetX_32GF"], use_ssld=use_ssld)
    return model


def RegNetY_200MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=36.44,
        w_0=24,
        w_m=2.49,
        d=13,
        group_w=8,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_200MF"], use_ssld=use_ssld)
    return model


def RegNetY_400MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=27.89,
        w_0=48,
        w_m=2.09,
        d=16,
        group_w=8,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_400MF"], use_ssld=use_ssld)
    return model


def RegNetY_600MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=32.54,
        w_0=48,
        w_m=2.32,
        d=15,
        group_w=16,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_600MF"], use_ssld=use_ssld)
    return model


def RegNetY_800MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=38.84,
        w_0=56,
        w_m=2.4,
        d=14,
        group_w=16,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_800MF"], use_ssld=use_ssld)
    return model


def RegNetY_1600MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=20.71,
        w_0=48,
        w_m=2.65,
        d=27,
        group_w=24,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_1600MF"], use_ssld=use_ssld)
    return model


def RegNetY_3200MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=42.63,
        w_0=80,
        w_m=2.66,
        d=21,
        group_w=24,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_3200MF"], use_ssld=use_ssld)
    return model


def RegNetY_4GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=31.41,
        w_0=96,
        w_m=2.24,
        d=22,
        group_w=64,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_4GF"], use_ssld=use_ssld)
    return model


def RegNetY_6400MF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=33.22,
        w_0=112,
        w_m=2.27,
        d=25,
        group_w=72,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_6400MF"], use_ssld=use_ssld)
    return model


def RegNetY_8GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=76.82,
        w_0=192,
        w_m=2.19,
        d=17,
        group_w=56,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_8GF"], use_ssld=use_ssld)
    return model


def RegNetY_12GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=73.36,
        w_0=168,
        w_m=2.37,
        d=19,
        group_w=112,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_12GF"], use_ssld=use_ssld)
    return model


def RegNetY_16GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=106.23,
        w_0=200,
        w_m=2.48,
        d=18,
        group_w=112,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_16GF"], use_ssld=use_ssld)
    return model


def RegNetY_32GF(pretrained=False, use_ssld=False, **kwargs):
    model = RegNet(
        w_a=115.89,
        w_0=232,
        w_m=2.53,
        d=20,
        group_w=232,
        bot_mul=1.0,
        q=8,
        se_on=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["RegNetY_32GF"], use_ssld=use_ssld)
    return model
