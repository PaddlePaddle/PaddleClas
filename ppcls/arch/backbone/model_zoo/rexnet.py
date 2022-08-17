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

# reference: https://arxiv.org/abs/2007.00992

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from math import ceil

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "ReXNet_1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_0_pretrained.pdparams",
    "ReXNet_1_3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_3_pretrained.pdparams",
    "ReXNet_1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_1_5_pretrained.pdparams",
    "ReXNet_2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_2_0_pretrained.pdparams",
    "ReXNet_3_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ReXNet_3_0_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


def conv_bn_act(out,
                in_channels,
                channels,
                kernel=1,
                stride=1,
                pad=0,
                num_group=1,
                active=True,
                relu6=False):
    out.append(
        nn.Conv2D(
            in_channels,
            channels,
            kernel,
            stride,
            pad,
            groups=num_group,
            bias_attr=False))
    out.append(nn.BatchNorm2D(channels))
    if active:
        out.append(nn.ReLU6() if relu6 else nn.ReLU())


def conv_bn_swish(out,
                  in_channels,
                  channels,
                  kernel=1,
                  stride=1,
                  pad=0,
                  num_group=1):
    out.append(
        nn.Conv2D(
            in_channels,
            channels,
            kernel,
            stride,
            pad,
            groups=num_group,
            bias_attr=False))
    out.append(nn.BatchNorm2D(channels))
    out.append(nn.Swish())


class SE(nn.Layer):
    def __init__(self, in_channels, channels, se_ratio=12):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc = nn.Sequential(
            nn.Conv2D(
                in_channels, channels // se_ratio, kernel_size=1, padding=0),
            nn.BatchNorm2D(channels // se_ratio),
            nn.ReLU(),
            nn.Conv2D(
                channels // se_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class LinearBottleneck(nn.Layer):
    def __init__(self,
                 in_channels,
                 channels,
                 t,
                 stride,
                 use_se=True,
                 se_ratio=12,
                 **kwargs):
        super(LinearBottleneck, self).__init__(**kwargs)
        self.use_shortcut = stride == 1 and in_channels <= channels
        self.in_channels = in_channels
        self.out_channels = channels

        out = []
        if t != 1:
            dw_channels = in_channels * t
            conv_bn_swish(out, in_channels=in_channels, channels=dw_channels)
        else:
            dw_channels = in_channels

        conv_bn_act(
            out,
            in_channels=dw_channels,
            channels=dw_channels,
            kernel=3,
            stride=stride,
            pad=1,
            num_group=dw_channels,
            active=False)

        if use_se:
            out.append(SE(dw_channels, dw_channels, se_ratio))

        out.append(nn.ReLU6())
        conv_bn_act(
            out,
            in_channels=dw_channels,
            channels=channels,
            active=False,
            relu6=True)
        self.out = nn.Sequential(*out)

    def forward(self, x):
        out = self.out(x)
        if self.use_shortcut:
            out[:, 0:self.in_channels] += x

        return out


class ReXNetV1(nn.Layer):
    def __init__(self,
                 input_ch=16,
                 final_ch=180,
                 width_mult=1.0,
                 depth_mult=1.0,
                 class_num=1000,
                 use_se=True,
                 se_ratio=12,
                 dropout_ratio=0.2,
                 bn_momentum=0.9):
        super(ReXNetV1, self).__init__()

        layers = [1, 2, 2, 3, 3, 5]
        strides = [1, 2, 2, 2, 1, 2]
        use_ses = [False, False, True, True, True, True]

        layers = [ceil(element * depth_mult) for element in layers]
        strides = sum([[element] + [1] * (layers[idx] - 1)
                       for idx, element in enumerate(strides)], [])
        if use_se:
            use_ses = sum([[element] * layers[idx]
                           for idx, element in enumerate(use_ses)], [])
        else:
            use_ses = [False] * sum(layers[:])
        ts = [1] * layers[0] + [6] * sum(layers[1:])

        self.depth = sum(layers[:]) * 3
        stem_channel = 32 / width_mult if width_mult < 1.0 else 32
        inplanes = input_ch / width_mult if width_mult < 1.0 else input_ch

        features = []
        in_channels_group = []
        channels_group = []

        # The following channel configuration is a simple instance to make each layer become an expand layer.
        for i in range(self.depth // 3):
            if i == 0:
                in_channels_group.append(int(round(stem_channel * width_mult)))
                channels_group.append(int(round(inplanes * width_mult)))
            else:
                in_channels_group.append(int(round(inplanes * width_mult)))
                inplanes += final_ch / (self.depth // 3 * 1.0)
                channels_group.append(int(round(inplanes * width_mult)))

        conv_bn_swish(
            features,
            3,
            int(round(stem_channel * width_mult)),
            kernel=3,
            stride=2,
            pad=1)

        for block_idx, (in_c, c, t, s, se) in enumerate(
                zip(in_channels_group, channels_group, ts, strides, use_ses)):
            features.append(
                LinearBottleneck(
                    in_channels=in_c,
                    channels=c,
                    t=t,
                    stride=s,
                    use_se=se,
                    se_ratio=se_ratio))

        pen_channels = int(1280 * width_mult)
        conv_bn_swish(features, c, pen_channels)

        features.append(nn.AdaptiveAvgPool2D(1))
        self.features = nn.Sequential(*features)
        self.output = nn.Sequential(
            nn.Dropout(dropout_ratio),
            nn.Conv2D(
                pen_channels, class_num, 1, bias_attr=True))

    def forward(self, x):
        x = self.features(x)
        x = self.output(x).squeeze(axis=-1).squeeze(axis=-1)
        return x


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


def ReXNet_1_0(pretrained=False, use_ssld=False, **kwargs):
    model = ReXNetV1(width_mult=1.0, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ReXNet_1_0"], use_ssld=use_ssld)
    return model


def ReXNet_1_3(pretrained=False, use_ssld=False, **kwargs):
    model = ReXNetV1(width_mult=1.3, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ReXNet_1_3"], use_ssld=use_ssld)
    return model


def ReXNet_1_5(pretrained=False, use_ssld=False, **kwargs):
    model = ReXNetV1(width_mult=1.5, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ReXNet_1_5"], use_ssld=use_ssld)
    return model


def ReXNet_2_0(pretrained=False, use_ssld=False, **kwargs):
    model = ReXNetV1(width_mult=2.0, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ReXNet_2_0"], use_ssld=use_ssld)
    return model


def ReXNet_3_0(pretrained=False, use_ssld=False, **kwargs):
    model = ReXNetV1(width_mult=3.0, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ReXNet_3_0"], use_ssld=use_ssld)
    return model
