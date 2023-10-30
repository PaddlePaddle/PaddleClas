# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
#
# Code was heavily based on https://github.com/liyunsheng13/micronet
# reference: https://arxiv.org/pdf/2108.05894

import math

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MicroNet_M0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MicroNet_M0_pretrained.pdparams",
    "MicroNet_M1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MicroNet_M1_pretrained.pdparams",
    "MicroNet_M2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MicroNet_M2_pretrained.pdparams",
    "MicroNet_M3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MicroNet_M3_pretrained.pdparams",
}

__all__ = MODEL_URLS.keys()

NET_CONFIG = {
    "msnx_dy6_exp4_4M_221": [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4,y1,y2,y3,r
        [2, 1, 8, 3, 2, 2, 0, 4, 8, 2, 2, 2, 0, 1,
         1],  # 6  ->12(0,0)  ->24  ->8(4,2)   ->8
        [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 2, 1,
         1],  # 8  ->16(0,0)  ->32  ->16(4,4)  ->12
        [2, 1, 16, 5, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1,
         1],  # 16 ->32(0,0)  ->64  ->16(8,2)  ->16
        [1, 1, 32, 5, 1, 4, 4, 4, 32, 4, 4, 2, 2, 1,
         1],  # 16 ->16(2,8)  ->96  ->32(8,4)  ->32
        [2, 1, 64, 5, 1, 4, 8, 8, 64, 8, 8, 2, 2, 1,
         1],  # 32 ->32(2,16) ->192 ->64(12,4) ->64
        [1, 1, 96, 3, 1, 4, 8, 8, 96, 8, 8, 2, 2, 1,
         2],  # 64 ->64(3,16) ->384 ->96(16,6) ->96
        [1, 1, 384, 3, 1, 4, 12, 12, 0, 0, 0, 2, 2, 1,
         2],  # 96 ->96(4,24) ->384
    ],
    "msnx_dy6_exp6_6M_221": [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1, 8, 3, 2, 2, 0, 6, 8, 2, 2, 2, 0, 1,
         1],  # 6  ->12(0,0)  ->24  ->8(4,2)   ->8
        [2, 1, 16, 3, 2, 2, 0, 8, 16, 4, 4, 2, 2, 1,
         1],  # 8  ->16(0,0)  ->32  ->16(4,4)  ->16
        [2, 1, 16, 5, 2, 2, 0, 16, 16, 4, 4, 2, 2, 1,
         1],  # 16 ->32(0,0)  ->64  ->16(8,2)  ->16
        [1, 1, 32, 5, 1, 6, 4, 4, 32, 4, 4, 2, 2, 1,
         1],  # 16 ->16(2,8)  ->96  ->32(8,4)  ->32
        [2, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1,
         1],  # 32 ->32(2,16) ->192 ->64(12,4) ->64
        [1, 1, 96, 3, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1,
         2],  # 64 ->64(3,16) ->384 ->96(16,6) ->96
        [1, 1, 576, 3, 1, 6, 12, 12, 0, 0, 0, 2, 2, 1,
         2],  # 96 ->96(4,24) ->576
    ],
    "msnx_dy9_exp6_12M_221": [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1, 12, 3, 2, 2, 0, 8, 12, 4, 4, 2, 0, 1,
         1],  # 8   ->16(0,0)   ->32  ->12(4,3)   ->12
        [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 2, 2, 1,
         1],  # 12  ->24(0,0)   ->48  ->16(8,2)   ->16
        [1, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 2, 2, 1,
         1],  # 16  ->16(0,0)   ->64  ->24(8,3)   ->24
        [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 2, 2, 1,
         1],  # 24  ->24(2,12)  ->144 ->32(16,2)  ->32
        [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 2, 2, 1,
         2],  # 32  ->32(2,16)  ->192 ->32(16,2)  ->32
        [1, 1, 64, 5, 1, 6, 8, 8, 64, 8, 8, 2, 2, 1,
         2],  # 32  ->32(2,16)  ->192 ->64(12,4)  ->64
        [2, 1, 96, 5, 1, 6, 8, 8, 96, 8, 8, 2, 2, 1,
         2],  # 64  ->64(4,12)  ->384 ->96(16,5)  ->96
        [1, 1, 128, 3, 1, 6, 12, 12, 128, 8, 8, 2, 2, 1,
         2],  # 96  ->96(5,16)  ->576 ->128(16,8) ->128
        [1, 1, 768, 3, 1, 6, 16, 16, 0, 0, 0, 2, 2, 1,
         2],  # 128 ->128(4,32) ->768
    ],
    "msnx_dy12_exp6_20M_020": [
        #s, n,  c, ks, c1, c2, g1, g2, c3, g3, g4
        [2, 1, 16, 3, 2, 2, 0, 12, 16, 4, 4, 0, 2, 0,
         1],  # 12  ->24(0,0)   ->48  ->16(8,2)   ->16
        [2, 1, 24, 3, 2, 2, 0, 16, 24, 4, 4, 0, 2, 0,
         1],  # 16  ->32(0,0)   ->64  ->24(8,3)   ->24
        [1, 1, 24, 3, 2, 2, 0, 24, 24, 4, 4, 0, 2, 0,
         1],  # 24  ->48(0,0)   ->96  ->24(8,3)   ->24
        [2, 1, 32, 5, 1, 6, 6, 6, 32, 4, 4, 0, 2, 0,
         1],  # 24  ->24(2,12)  ->144 ->32(16,2)  ->32
        [1, 1, 32, 5, 1, 6, 8, 8, 32, 4, 4, 0, 2, 0,
         2],  # 32  ->32(2,16)  ->192 ->32(16,2)  ->32
        [1, 1, 64, 5, 1, 6, 8, 8, 48, 8, 8, 0, 2, 0,
         2],  # 32  ->32(2,16)  ->192 ->48(12,4)  ->64
        [1, 1, 80, 5, 1, 6, 8, 8, 80, 8, 8, 0, 2, 0,
         2],  # 48  ->48(3,16)  ->288 ->80(16,5)  ->80
        [1, 1, 80, 5, 1, 6, 10, 10, 80, 8, 8, 0, 2, 0,
         2],  # 80  ->80(4,20)  ->480 ->80(20,4)  ->80
        [2, 1, 120, 5, 1, 6, 10, 10, 120, 10, 10, 0, 2, 0,
         2],  # 80  ->80(4,20)  ->480 ->128(16,8) ->120
        [1, 1, 120, 5, 1, 6, 12, 12, 120, 10, 10, 0, 2, 0,
         2],  # 120 ->128(4,32) ->720 ->128(32,4) ->120
        [1, 1, 144, 3, 1, 6, 12, 12, 144, 12, 12, 0, 2, 0,
         2],  # 120 ->128(4,32) ->720 ->160(32,5) ->144
        [1, 1, 864, 3, 1, 6, 12, 12, 0, 0, 0, 0, 2, 0,
         2],  # 144 ->144(5,32) ->864
    ],
}

ACTIVATION_CONFIG = {
    "msnx_dy6_exp4_4M_221": {
        "act_max": 2.0,
        "reduction": 8,
        "init_ab3": [1.0, 0.0],
        "init_a": [1.0, 1.0],
        "init_b": [0.0, 0.0],
    },
    "msnx_dy6_exp6_6M_221": {
        "act_max": 2.0,
        "reduction": 8,
        "init_ab3": [1.0, 0.0],
        "init_a": [1.0, 1.0],
        "init_b": [0.0, 0.0],
    },
    "msnx_dy9_exp6_12M_221": {
        "act_max": 2.0,
        "reduction": 8,
        "init_ab3": [1.0, 0.0],
        "init_a": [1.0, 1.0],
        "init_b": [0.0, 0.0],
    },
    "msnx_dy12_exp6_20M_020": {
        "act_max": 2.0,
        "reduction": 8,
        "init_ab3": [1.0, 0.0],
        "init_a": [1.0, 0.5],
        "init_b": [0.0, 0.5],
    },
}


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class MaxGroupPooling(nn.Layer):
    def __init__(self, channel_per_group=2):
        super().__init__()
        self.channel_per_group = channel_per_group

    def forward(self, x):
        if self.channel_per_group == 1:
            return x

        # max op
        b, c, h, w = x.shape

        # reshape
        y = x.reshape([b, c // self.channel_per_group, -1, h, w])
        out, _ = paddle.max(y, axis=2)
        return out


class SwishLinear(nn.Layer):
    def __init__(self, inp, oup):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(inp, oup), nn.BatchNorm1D(oup), nn.Hardswish())

    def forward(self, x):
        return self.linear(x)


class StemLayer(nn.Layer):
    def __init__(self, inp, oup, stride, groups=(4, 4)):
        super().__init__()
        g1, g2 = groups
        self.stem = nn.Sequential(
            SpatialSepConvSF(inp, groups, 3, stride),
            MaxGroupPooling(2) if g1 * g2 == 2 * oup else nn.ReLU6())

    def forward(self, x):
        out = self.stem(x)
        return out


class GroupConv(nn.Layer):
    def __init__(self, inp, oup, groups=2):
        super().__init__()
        self.inp = inp
        self.oup = oup
        self.groups = groups
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp, oup, 1, groups=self.groups[0], bias_attr=False),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelShuffle(nn.Layer):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        b, c, h, w = x.shape

        channels_per_group = c // self.groups

        # reshape
        x = x.reshape([b, self.groups, channels_per_group, h, w])
        x = x.transpose([0, 2, 1, 3, 4])
        out = x.reshape([b, c, h, w])

        return out


class SpatialSepConvSF(nn.Layer):
    def __init__(self, inp, oups, kernel_size, stride):
        super().__init__()

        oup1, oup2 = oups
        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                oup1, (kernel_size, 1), (stride, 1), (kernel_size // 2, 0),
                groups=1,
                bias_attr=False),
            nn.BatchNorm2D(oup1),
            nn.Conv2D(
                oup1,
                oup1 * oup2, (1, kernel_size), (1, stride),
                (0, kernel_size // 2),
                groups=oup1,
                bias_attr=False),
            nn.BatchNorm2D(oup1 * oup2),
            ChannelShuffle(oup1))

    def forward(self, x):
        out = self.conv(x)
        return out


class DepthSpatialSepConv(nn.Layer):
    def __init__(self, inp, expand, kernel_size, stride):
        super().__init__()

        exp1, exp2 = expand
        hidden_dim = inp * exp1
        oup = inp * exp1 * exp2

        self.conv = nn.Sequential(
            nn.Conv2D(
                inp,
                inp * exp1, (kernel_size, 1), (stride, 1),
                (kernel_size // 2, 0),
                groups=inp,
                bias_attr=False),
            nn.BatchNorm2D(inp * exp1),
            nn.Conv2D(
                hidden_dim,
                oup, (1, kernel_size), (1, stride), (0, kernel_size // 2),
                groups=hidden_dim,
                bias_attr=False),
            nn.BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)
        return out


class DYShiftMax(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 reduction=4,
                 act_max=1.0,
                 act_relu=True,
                 init_a=[0.0, 0.0],
                 init_b=[0.0, 0.0],
                 relu_before_pool=False,
                 g=None,
                 expansion=False):
        super().__init__()
        self.oup = oup
        self.act_max = act_max * 2
        self.act_relu = act_relu
        self.avg_pool = nn.Sequential(nn.ReLU() if relu_before_pool == True
                                      else nn.Identity(),
                                      nn.AdaptiveAvgPool2D(1))

        self.exp = 4 if act_relu else 2
        self.init_a = init_a
        self.init_b = init_b

        # determine squeeze
        squeeze = _make_divisible(inp // reduction, 4)
        if squeeze < 4:
            squeeze = 4

        self.fc = nn.Sequential(
            nn.Linear(inp, squeeze),
            nn.ReLU(), nn.Linear(squeeze, oup * self.exp), nn.Hardsigmoid())
        if g is None:
            g = 1
        self.g = g[1]
        if self.g != 1 and expansion:
            self.g = inp // self.g
        self.gc = inp // self.g
        index = paddle.to_tensor(list(range(inp))).reshape([1, inp, 1, 1])
        index = index.reshape([1, self.g, self.gc, 1, 1])
        indexgs = paddle.split(index, [1, self.g - 1], axis=1)
        indexgs = paddle.concat((indexgs[1], indexgs[0]), axis=1)
        indexs = paddle.split(indexgs, [1, self.gc - 1], axis=2)
        indexs = paddle.concat((indexs[1], indexs[0]), axis=2)
        self.index = indexs.reshape([inp]).astype(paddle.int64)
        self.expansion = expansion

    def forward(self, x):
        x_in = x
        x_out = x

        b, c, _, _ = x_in.shape
        y = self.avg_pool(x_in).reshape([b, c])
        y = self.fc(y).reshape([b, self.oup * self.exp, 1, 1])
        y = (y - 0.5) * self.act_max

        n2, c2, h2, w2 = x_out.shape
        x2 = paddle.index_select(x_out, self.index, axis=1)

        if self.exp == 4:
            a1, b1, a2, b2 = paddle.split(y, 4, axis=1)

            a1 = a1 + self.init_a[0]
            a2 = a2 + self.init_a[1]

            b1 = b1 + self.init_b[0]
            b2 = b2 + self.init_b[1]

            z1 = x_out * a1 + x2 * b1
            z2 = x_out * a2 + x2 * b2

            out = paddle.maximum(z1, z2)

        elif self.exp == 2:
            a1, b1 = paddle.split(y, 2, axis=1)
            a1 = a1 + self.init_a[0]
            b1 = b1 + self.init_b[0]
            out = x_out * a1 + x2 * b1

        return out


class DYMicroBlock(nn.Layer):
    def __init__(self,
                 inp,
                 oup,
                 kernel_size=3,
                 stride=1,
                 ch_exp=(2, 2),
                 ch_per_group=4,
                 groups_1x1=(1, 1),
                 dy=[0, 0, 0],
                 ratio=1.0,
                 activation_cfg=None):
        super().__init__()

        self.identity = stride == 1 and inp == oup

        y1, y2, y3 = dy
        act_max = activation_cfg["act_max"]
        act_reduction = activation_cfg["reduction"] * ratio
        init_a = activation_cfg["init_a"]
        init_b = activation_cfg["init_b"]
        init_ab3 = activation_cfg["init_ab3"]

        t1 = ch_exp
        gs1 = ch_per_group
        hidden_fft, g1, g2 = groups_1x1

        hidden_dim1 = inp * t1[0]
        hidden_dim2 = inp * t1[0] * t1[1]

        if gs1[0] == 0:
            self.layers = nn.Sequential(
                DepthSpatialSepConv(inp, t1, kernel_size, stride),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]),
                ChannelShuffle(hidden_dim2 // 2) if y2 != 0 else nn.Identity(),
                GroupConv(hidden_dim2, oup, (g1, g2)),
                DYShiftMax(
                    oup,
                    oup,
                    act_max=act_max,
                    act_relu=False,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction // 2,
                    init_b=[init_ab3[1], 0.0],
                    g=(g1, g2),
                    expansion=False) if y3 > 0 else nn.Identity(),
                ChannelShuffle(g2),
                ChannelShuffle(oup // 2)
                if oup % 2 == 0 and y3 != 0 else nn.Identity())
        elif g2 == 0:
            self.layers = nn.Sequential(
                GroupConv(inp, hidden_dim2, gs1),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=act_max,
                    act_relu=False,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g=gs1,
                    expansion=False) if y3 > 0 else nn.Identity())
        else:
            self.layers = nn.Sequential(
                GroupConv(inp, hidden_dim2, gs1),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=act_max,
                    act_relu=True if y1 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=False) if y1 > 0 else nn.ReLU6(),
                ChannelShuffle(gs1[1]),
                DepthSpatialSepConv(hidden_dim2, (1, 1), kernel_size, stride),
                nn.Identity(),
                DYShiftMax(
                    hidden_dim2,
                    hidden_dim2,
                    act_max=act_max,
                    act_relu=True if y2 == 2 else False,
                    init_a=init_a,
                    reduction=act_reduction,
                    init_b=init_b,
                    g=gs1,
                    expansion=True) if y2 > 0 else nn.ReLU6(),
                ChannelShuffle(hidden_dim2 // 4)
                if y1 != 0 and y2 != 0 else nn.Identity()
                if y1 == 0 and y2 == 0 else ChannelShuffle(hidden_dim2 // 2),
                GroupConv(hidden_dim2, oup, (g1, g2)),
                DYShiftMax(
                    oup,
                    oup,
                    act_max=act_max,
                    act_relu=False,
                    init_a=[init_ab3[0], 0.0],
                    reduction=act_reduction // 2
                    if oup < hidden_dim2 else act_reduction,
                    init_b=[init_ab3[1], 0.0],
                    g=(g1, g2),
                    expansion=False) if y3 > 0 else nn.Identity(),
                ChannelShuffle(g2),
                ChannelShuffle(oup // 2) if y3 != 0 else nn.Identity())

    def forward(self, x):
        out = self.layers(x)
        if self.identity:
            out = out + x
        return out


class MicroNet(nn.Layer):
    def __init__(self,
                 net_cfg,
                 activation_cfg,
                 input_size=224,
                 class_num=1000,
                 stem_ch=16,
                 stem_groups=[4, 8],
                 out_ch=1024,
                 dropout_rate=0.0):
        super().__init__()

        # building first layer
        assert input_size % 32 == 0
        input_channel = stem_ch
        layers = [StemLayer(3, input_channel, stride=2, groups=stem_groups)]

        for s, n, c, ks, c1, c2, g1, g2, c3, g3, g4, y1, y2, y3, r in net_cfg:
            for i in range(n):
                layers.append(
                    DYMicroBlock(
                        input_channel,
                        c,
                        kernel_size=ks,
                        stride=s if i == 0 else 1,
                        ch_exp=(c1, c2),
                        ch_per_group=(g1, g2),
                        groups_1x1=(c3, g3, g4),
                        dy=[y1, y2, y3],
                        ratio=r,
                        activation_cfg=activation_cfg))
                input_channel = c
        self.features = nn.Sequential(*layers)

        self.avgpool = nn.Sequential(nn.ReLU6(),
                                     nn.AdaptiveAvgPool2D(1), nn.Hardswish())

        # building last several layers
        self.classifier = nn.Sequential(
            SwishLinear(input_channel, out_ch),
            nn.Dropout(dropout_rate), SwishLinear(out_ch, class_num))

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2D):
            n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            nn.initializer.Normal(std=math.sqrt(2. / n))(m.weight)
        elif isinstance(m, nn.Linear):
            nn.initializer.Normal(std=0.01)(m.weight)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x.flatten(1))
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def MicroNet_M0(pretrained=False, use_ssld=False, **kwargs):
    model = MicroNet(
        NET_CONFIG["msnx_dy6_exp4_4M_221"],
        ACTIVATION_CONFIG["msnx_dy6_exp4_4M_221"],
        stem_ch=4,
        stem_groups=[2, 2],
        out_ch=640,
        dropout_rate=0.05,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MicroNet_M0"], use_ssld)
    return model


def MicroNet_M1(pretrained=False, use_ssld=False, **kwargs):
    model = MicroNet(
        NET_CONFIG["msnx_dy6_exp6_6M_221"],
        ACTIVATION_CONFIG["msnx_dy6_exp6_6M_221"],
        stem_ch=6,
        stem_groups=[3, 2],
        out_ch=960,
        dropout_rate=0.05,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MicroNet_M1"], use_ssld)
    return model


def MicroNet_M2(pretrained=False, use_ssld=False, **kwargs):
    model = MicroNet(
        NET_CONFIG["msnx_dy9_exp6_12M_221"],
        ACTIVATION_CONFIG["msnx_dy9_exp6_12M_221"],
        stem_ch=8,
        stem_groups=[4, 2],
        out_ch=1024,
        dropout_rate=0.1,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MicroNet_M2"], use_ssld)
    return model


def MicroNet_M3(pretrained=False, use_ssld=False, **kwargs):
    model = MicroNet(
        NET_CONFIG["msnx_dy12_exp6_20M_020"],
        ACTIVATION_CONFIG["msnx_dy12_exp6_20M_020"],
        stem_ch=12,
        stem_groups=[4, 3],
        out_ch=1024,
        dropout_rate=0.1,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MicroNet_M3"], use_ssld)
    return model
