# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
Origin author: wujiyang
Modify from: https://github.com/wujiyang/Face_Pytorch/blob/master/backbone/mobilefacenet.py
'''

import paddle
from paddle import nn
import math

__all__ = ['MobileFaceNet']

MobileFaceNet_BottleNeck_Setting = [
    # t, c , n ,s
    [2, 64, 5, 2],
    [4, 128, 1, 2],
    [2, 128, 6, 1],
    [4, 128, 1, 2],
    [2, 128, 2, 1]
]


class BottleNeck(nn.Layer):
    def __init__(self, inp, oup, stride, expansion, data_format="NCHW"):
        super().__init__()
        self.connect = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # 1*1 conv
            nn.Conv2D(
                inp, inp * expansion, 1, 1, 0, bias_attr=False, data_format=data_format),
            nn.BatchNorm2D(inp * expansion, data_format=data_format),
            nn.PReLU(inp * expansion, data_format=data_format),

            # 3*3 depth wise conv
            nn.Conv2D(
                inp * expansion,
                inp * expansion,
                3,
                stride,
                1,
                groups=inp * expansion,
                bias_attr=False,
                data_format=data_format
            ),
            nn.BatchNorm2D(inp * expansion, data_format=data_format),
            nn.PReLU(inp * expansion, data_format=data_format),

            # 1*1 conv
            nn.Conv2D(
                inp * expansion, oup, 1, 1, 0, bias_attr=False, data_format=data_format),
            nn.BatchNorm2D(oup, data_format=data_format), )

    def forward(self, x):
        if self.connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvBlock(nn.Layer):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False, data_format="NCHW"):
        super().__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2D(
                inp, oup, k, s, p, groups=inp, bias_attr=False, data_format=data_format)
        else:
            self.conv = nn.Conv2D(inp, oup, k, s, p, bias_attr=False, data_format=data_format)

        self.bn = nn.BatchNorm2D(oup, data_format=data_format)
        if not linear:
            self.prelu = nn.PReLU(oup, data_format=data_format)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)


class mobilefacenet(nn.Layer):
    def __init__(self,
                 feature_dim=128,
                 bottleneck_setting=MobileFaceNet_BottleNeck_Setting,
                 data_format="NCHW",
                 **args):
        super().__init__()
        self.data_format = data_format
        
        self.conv1 = ConvBlock(3, 64, 3, 2, 1, data_format=data_format)
        self.dw_conv1 = ConvBlock(64, 64, 3, 1, 1, dw=True, data_format=data_format)

        self.cur_channel = 64
        block = BottleNeck
        self.blocks = self._make_layer(block, bottleneck_setting)

        self.conv2 = ConvBlock(128, 512, 1, 1, 0, data_format=data_format)
        self.linear7 = ConvBlock(512, 512, 7, 1, 0, dw=True, linear=True, data_format=data_format)
        self.linear1 = ConvBlock(512, feature_dim, 1, 1, 0, linear=True, data_format=data_format)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                # ks * ks * out_ch
                n = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype=m.weight.dtype,
                    default_initializer=nn.initializer.Normal(
                        mean=0.0, std=math.sqrt(2.0 / n)))

            elif isinstance(m, (nn.BatchNorm, nn.BatchNorm2D, nn.GroupNorm)):
                m.weight = paddle.create_parameter(
                    shape=m.weight.shape,
                    dtype=m.weight.dtype,
                    default_initializer=nn.initializer.Constant(value=1.0))
                m.bias = paddle.create_parameter(
                    shape=m.bias.shape,
                    dtype=m.bias.dtype,
                    default_initializer=nn.initializer.Constant(value=0.0))

    def _make_layer(self, block, setting):
        layers = []
        for t, c, n, s in setting:
            for i in range(n):
                if i == 0:
                    layers.append(block(self.cur_channel, c, s, t, data_format=self.data_format))
                else:
                    layers.append(block(self.cur_channel, c, 1, t, data_format=self.data_format))
                self.cur_channel = c

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.data_format == "NHWC":
            x = paddle.transpose(x, [0, 2, 3, 1])
            x.stop_gradient = True
        x = self.conv1(x)
        x = self.dw_conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.linear7(x)
        x = self.linear1(x)
        if self.data_format == "NHWC":
            x = paddle.transpose(x, [0, 3, 1, 2])
        x = x.reshape([x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]])
        return x


def MobileFaceNet(num_features=128, **args):
    model = mobilefacenet(feature_dim=num_features, **args)
    return model