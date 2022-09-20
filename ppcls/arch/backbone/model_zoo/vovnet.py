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
# Code was heavily based on https://github.com/stigma0617/VoVNet.pytorch/blob/master/models_vovnet/vovnet.py

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, KaimingNormal

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "VoVNet27_slim": "",
    "VoVNet39": "",
    "VoVNet57": "",
}


def conv3x3(in_channels,
            out_channels,
            stride=1,
            groups=1,
            kernel_size=3,
            padding=1):
    """3x3 convolution with padding"""
    return nn.Sequential(
        nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False),
        nn.BatchNorm2D(out_channels),
        nn.ReLU())


def conv1x1(in_channels,
            out_channels,
            stride=1,
            groups=1,
            kernel_size=1,
            padding=0):
    """1x1 convolution"""
    return nn.Sequential(
        nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=False),
        nn.BatchNorm2D(out_channels),
        nn.ReLU())


class _OSA_module(nn.Layer):
    def __init__(self,
                 in_ch,
                 stage_ch,
                 concat_ch,
                 layer_per_block,
                 identity=False):
        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.LayerList()
        in_channel = in_ch
        for i in range(layer_per_block):
            self.layers.append(conv3x3(in_channel, stage_ch))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = conv1x1(in_channel, concat_ch)

    def forward(self, x):
        identity_feat = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = paddle.concat(output, axis=1)
        xt = self.concat(x)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):
    def __init__(self, in_ch, stage_ch, concat_ch, block_per_stage,
                 layer_per_block, stage_num):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_sublayer(
                'Pooling',
                nn.MaxPool2D(
                    kernel_size=3, stride=2, ceil_mode=True))

        module_name = f'OSA{stage_num}_1'
        self.add_sublayer(module_name,
                          _OSA_module(in_ch, stage_ch, concat_ch,
                                      layer_per_block))
        for i in range(block_per_stage - 1):
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_sublayer(
                module_name,
                _OSA_module(
                    concat_ch,
                    stage_ch,
                    concat_ch,
                    layer_per_block,
                    identity=True))


class VoVNet(nn.Layer):
    r""" VoVNet
        A PaddlePaddle impl of : `An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection`
          https://arxiv.org/abs/1904.09730

    Args:
        config_stage_ch (tuple(int)): Output channels at each stage.
        config_concat_ch (int): Output channels of concatenations.
        block_per_stage (float): Stochastic depth rate. Default: 0.
        layer_per_block (float): Num of layers for each block. Default: 5.
        class_num (int): Number of classes for classification head. Default: 1000
    """

    def __init__(self,
                 config_stage_ch,
                 config_concat_ch,
                 block_per_stage,
                 layer_per_block,
                 class_num=1000):
        super(VoVNet, self).__init__()

        self.stem = nn.Sequential(
            conv3x3(3, 64, 2), conv3x3(64, 64, 1), conv3x3(64, 128, 2))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        self.stage_names = []
        for i in range(4):  # num_stages
            name = 'stage%d' % (i + 2)
            self.stage_names.append(name)
            self.add_sublayer(
                name,
                _OSA_stage(in_ch_list[i], config_stage_ch[i],
                           config_concat_ch[i], block_per_stage[i],
                           layer_per_block, i + 2))

        self.classifier = nn.Linear(config_concat_ch[-1], class_num)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                KaimingNormal()(m.weight)
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                Constant(value=1.)(m.weight)
                Constant(value=0.)(m.bias)
            elif isinstance(m, nn.Linear):
                Constant(value=0.)(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for name in self.stage_names:
            x = getattr(self, name)(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
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


def VoVNet57(pretrained=False, use_ssld=False, **kwargs):
    model = VoVNet([128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3],
                   5)

    _load_pretrained(
        pretrained, model, MODEL_URLS["VoVNet57"], use_ssld=use_ssld)
    return model


def VoVNet39(pretrained=False, use_ssld=False, **kwargs):
    model = VoVNet([128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3],
                   5)

    _load_pretrained(
        pretrained, model, MODEL_URLS["VoVNet39"], use_ssld=use_ssld)
    return model


def VoVNet27_slim(pretrained=False, use_ssld=False, **kwargs):
    model = VoVNet([128, 160, 192, 224], [256, 512, 768, 1024], [1, 1, 4, 3],
                   5)

    _load_pretrained(
        pretrained, model, MODEL_URLS["VoVNet27_slim"], use_ssld=use_ssld)
    return model
