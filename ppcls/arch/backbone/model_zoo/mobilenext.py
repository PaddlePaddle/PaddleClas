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

# Code was heavily based on https://github.com/zhoudaquan/rethinking_bottleneck_design
# reference: https://arxiv.org/abs/2007.02269

import math
from functools import partial

import paddle
import paddle.nn as nn
from paddle.regularizer import L2Decay

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "MobileNeXt_0.35": "",  # TODO
    "MobileNeXt_0.50": "",  # TODO
    "MobileNeXt_0.75": "",  # TODO
    "MobileNeXt_1.00": "",  # TODO
    "MobileNeXt_1.40": "",  # TODO
}

__all__ = list(MODEL_URLS.keys())

no_weight_decay_attr = paddle.ParamAttr(L2Decay(0))
BatchNorm2D = partial(
    nn.BatchNorm2D,
    weight_attr=no_weight_decay_attr,
    bias_attr=no_weight_decay_attr)
Conv2D = partial(nn.Conv2D, bias_attr=no_weight_decay_attr)
Linear = partial(nn.Linear, bias_attr=no_weight_decay_attr)


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        Conv2D(
            inp, oup, 3, stride, 1, bias_attr=False),
        BatchNorm2D(oup),
        nn.ReLU6())


class SGBlock(nn.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(SGBlock, self).__init__()
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio

        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6())
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                # pw-linear
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, stride, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                Conv2D(
                    inp, inp, 3, 1, 1, groups=inp, bias_attr=False),
                BatchNorm2D(inp),
                nn.ReLU6(),
                # pw
                Conv2D(
                    inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2D(hidden_dim),
                #nn.ReLU6(),
                # pw
                Conv2D(
                    hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2D(oup),
                nn.ReLU6(),
                # dw
                Conv2D(
                    oup, oup, 3, 1, 1, groups=oup, bias_attr=False),
                BatchNorm2D(oup))

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            if self.identity_div == 1:
                out = out + x
            else:
                shape = x.shape
                id_tensor = x[:, :shape[1] // self.identity_div, :, :]
                out[:, :shape[1] // self.identity_div, :, :] = \
                    out[:, :shape[1] // self.identity_div, :, :] + id_tensor

        return out


class MobileNeXt(nn.Layer):
    def __init__(self, class_num=1000, width_mult=1.00):
        super().__init__()

        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [2, 96, 1, 2],
            [6, 144, 1, 1],
            [6, 192, 3, 2],
            [6, 288, 3, 2],
            [6, 384, 4, 1],
            [6, 576, 4, 2],
            [6, 960, 3, 1],
            [6, 1280, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4
                                        if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SGBlock
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4
                                             if width_mult == 0.1 else 8)
            if c == 1280 and width_mult < 1:
                output_channel = 1280
            layers.append(
                block(input_channel, output_channel, s, t, n == 1 and s == 1))
            input_channel = output_channel
            for _ in range(n - 1):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel
        output_channel = _make_divisible(input_channel, 4)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            Linear(
                output_channel, class_num, bias_attr=no_weight_decay_attr))

        self.apply(self._initialize_weights)

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2D):
            n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            nn.initializer.Normal(std=math.sqrt(2. / n))(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            nn.initializer.Constant(1)(m.weight)
            nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Linear):
            nn.initializer.Normal(std=0.01)(m.weight)
            nn.initializer.Constant(0)(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
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


def MobileNeXt_35(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNeXt(width_mult=0.35, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNeXt_0.35"], use_ssld=use_ssld)
    return model


def MobileNeXt_50(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNeXt(width_mult=0.50, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNeXt_0.50"], use_ssld=use_ssld)
    return model


def MobileNeXt_75(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNeXt(width_mult=0.75, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNeXt_0.75"], use_ssld=use_ssld)
    return model


def MobileNeXt_100(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNeXt(width_mult=1.00, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNeXt_1.00"], use_ssld=use_ssld)
    return model


def MobileNeXt_140(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNeXt(width_mult=1.40, **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNeXt_1.40"], use_ssld=use_ssld)
    return model
