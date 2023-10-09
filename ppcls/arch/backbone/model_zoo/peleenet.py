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
# Code was heavily based on https://github.com/Robert-JunWang/PeleeNet
# reference: https://arxiv.org/pdf/1804.06882.pdf

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "PeleeNet":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/PeleeNet_pretrained.pdparams"
}

__all__ = MODEL_URLS.keys()

normal_ = lambda x, mean=0, std=1: Normal(mean, std)(x)
constant_ = lambda x, value=0: Constant(value)(x)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


class _DenseLayer(nn.Layer):
    def __init__(self, num_input_features, growth_rate, bottleneck_width,
                 drop_rate):
        super(_DenseLayer, self).__init__()

        growth_rate = int(growth_rate / 2)
        inter_channel = int(growth_rate * bottleneck_width / 4) * 4

        if inter_channel > num_input_features / 2:
            inter_channel = int(num_input_features / 8) * 4
            print('adjust inter_channel to ', inter_channel)

        self.branch1a = BasicConv2D(
            num_input_features, inter_channel, kernel_size=1)
        self.branch1b = BasicConv2D(
            inter_channel, growth_rate, kernel_size=3, padding=1)

        self.branch2a = BasicConv2D(
            num_input_features, inter_channel, kernel_size=1)
        self.branch2b = BasicConv2D(
            inter_channel, growth_rate, kernel_size=3, padding=1)
        self.branch2c = BasicConv2D(
            growth_rate, growth_rate, kernel_size=3, padding=1)

    def forward(self, x):
        branch1 = self.branch1a(x)
        branch1 = self.branch1b(branch1)

        branch2 = self.branch2a(x)
        branch2 = self.branch2b(branch2)
        branch2 = self.branch2c(branch2)

        return paddle.concat([x, branch1, branch2], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate,
                                growth_rate, bn_size, drop_rate)
            setattr(self, 'denselayer%d' % (i + 1), layer)


class _StemBlock(nn.Layer):
    def __init__(self, num_input_channels, num_init_features):
        super(_StemBlock, self).__init__()

        num_stem_features = int(num_init_features / 2)

        self.stem1 = BasicConv2D(
            num_input_channels,
            num_init_features,
            kernel_size=3,
            stride=2,
            padding=1)
        self.stem2a = BasicConv2D(
            num_init_features,
            num_stem_features,
            kernel_size=1,
            stride=1,
            padding=0)
        self.stem2b = BasicConv2D(
            num_stem_features,
            num_init_features,
            kernel_size=3,
            stride=2,
            padding=1)
        self.stem3 = BasicConv2D(
            2 * num_init_features,
            num_init_features,
            kernel_size=1,
            stride=1,
            padding=0)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.stem1(x)

        branch2 = self.stem2a(out)
        branch2 = self.stem2b(branch2)
        branch1 = self.pool(out)

        out = paddle.concat([branch1, branch2], 1)
        out = self.stem3(out)

        return out


class BasicConv2D(nn.Layer):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2D(
            in_channels, out_channels, bias_attr=False, **kwargs)
        self.norm = nn.BatchNorm2D(out_channels)
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            return F.relu(x)
        else:
            return x


class PeleeNetDY(nn.Layer):
    r"""PeleeNet model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf> and
     "Pelee: A Real-Time Object Detection System on Mobile Devices" <https://arxiv.org/pdf/1804.06882.pdf>`

    Args:
        growth_rate (int or list of 4 ints) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bottleneck_width (int or list of 4 ints) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        class_num (int) - number of classification classes
    """

    def __init__(self,
                 growth_rate=32,
                 block_config=[3, 4, 8, 6],
                 num_init_features=32,
                 bottleneck_width=[1, 2, 4, 4],
                 drop_rate=0.05,
                 class_num=1000):

        super(PeleeNetDY, self).__init__()

        self.features = nn.Sequential(* [('stemblock', _StemBlock(
            3, num_init_features)), ])

        if type(growth_rate) is list:
            growth_rates = growth_rate
            assert len(growth_rates) == 4, \
                'The growth rate must be the list and the size must be 4'
        else:
            growth_rates = [growth_rate] * 4

        if type(bottleneck_width) is list:
            bottleneck_widths = bottleneck_width
            assert len(bottleneck_widths) == 4, \
                'The bottleneck width must be the list and the size must be 4'
        else:
            bottleneck_widths = [bottleneck_width] * 4

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bottleneck_widths[i],
                growth_rate=growth_rates[i],
                drop_rate=drop_rate)
            setattr(self.features, 'denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rates[i]

            setattr(
                self.features,
                'transition%d' % (i + 1),
                BasicConv2D(
                    num_features,
                    num_features,
                    kernel_size=1,
                    stride=1,
                    padding=0))

            if i != len(block_config) - 1:
                setattr(
                    self.features,
                    'transition%d_pool' % (i + 1),
                    nn.AvgPool2D(
                        kernel_size=2, stride=2))
                num_features = num_features

        # Linear layer
        self.classifier = nn.Linear(num_features, class_num)
        self.drop_rate = drop_rate

        self.apply(self._initialize_weights)

    def forward(self, x):
        features = self.features(x)
        out = F.avg_pool2d(
            features, kernel_size=features.shape[2:4]).flatten(1)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.classifier(out)
        return out

    def _initialize_weights(self, m):
        if isinstance(m, nn.Conv2D):
            n = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            normal_(m.weight, std=math.sqrt(2. / n))
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            ones_(m.weight)
            zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            normal_(m.weight, std=0.01)
            zeros_(m.bias)


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


def PeleeNet(pretrained=False, use_ssld=False, **kwargs):
    model = PeleeNetDY(**kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PeleeNet"], use_ssld)
    return model
