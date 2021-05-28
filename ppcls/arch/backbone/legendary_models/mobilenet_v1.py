# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import, division, print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout, ReLU, Flatten
from paddle.nn import AdaptiveAvgPool2D
from paddle.nn.initializer import KaimingNormal
import math

from ppcls.arch.backbone.base.theseus_layer import TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain_from, load_dygraph_pretrain_from_url


MODEL_URLS = {
    "MobileNetV1_x0_25": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_25_pretrained.pdparams",
    "MobileNetV1_x0_5": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_5_vd_pretrained.pdparams",
    "MobileNetV1_x0_75": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_x0_75_pretrained.pdparams",
    "MobileNetV1": "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV1_pretrained.pdparams",
}

__all__ = MODEL_URLS.keys()
    
    
class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 num_channels,
                 filter_size,
                 num_filters,
                 stride,
                 padding,
                 num_groups=1):
        super(ConvBNLayer, self).__init__()

        self.conv = Conv2D(
            in_channels=num_channels,
            out_channels=num_filters,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal()),
            bias_attr=False)

        self.bn = BatchNorm(
            num_filters)
        
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class DepthwiseSeparable(TheseusLayer):
    def __init__(self,
                 num_channels,
                 num_filters1,
                 num_filters2,
                 num_groups,
                 stride,
                 scale):
        super(DepthwiseSeparable, self).__init__()

        self.depthwise_conv = ConvBNLayer(
            num_channels=num_channels,
            num_filters=int(num_filters1 * scale),
            filter_size=3,
            stride=stride,
            padding=1,
            num_groups=int(num_groups * scale))

        self.pointwise_conv = ConvBNLayer(
            num_channels=int(num_filters1 * scale),
            filter_size=1,
            num_filters=int(num_filters2 * scale),
            stride=1,
            padding=0)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class MobileNet(TheseusLayer):
    def __init__(self, scale=1.0, class_num=1000, pretrained=False):
        super(MobileNet, self).__init__()
        self.scale = scale
        self.pretrained = pretrained
        self.block_list = []

        self.conv = ConvBNLayer(
            num_channels=3,
            filter_size=3,
            num_filters=int(32 * scale),
            stride=2,
            padding=1)
        
        #num_channels, num_filters1, num_filters2, num_groups, stride
        self.cfg =   [[int(32 * scale),   32,   64,   32,   1],
                      [int(64 * scale),   64,   128,  64,   2],
                      [int(128 * scale),  128,  128,  128,  1],
                      [int(128 * scale),  128,  256,  128,  2],
                      [int(256 * scale),  256,  256,  256,  1],
                      [int(256 * scale),  256,  512,  256,  2],
                      [int(512 * scale),  512,  512,  512,  1],
                      [int(512 * scale),  512,  512,  512,  1],
                      [int(512 * scale),  512,  512,  512,  1],
                      [int(512 * scale),  512,  512,  512,  1],
                      [int(512 * scale),  512,  512,  512,  1],
                      [int(512 * scale),  512,  1024, 512,  2],
                      [int(1024 * scale), 1024, 1024, 1024, 1]]
        
        self.blocks = nn.Sequential(*[
                    DepthwiseSeparable(
                            num_channels=params[0],
                            num_filters1=params[1],
                            num_filters2=params[2],
                            num_groups=params[3],
                            stride=params[4],
                            scale=scale) for params in self.cfg])

        self.avg_pool = AdaptiveAvgPool2D(1)
        self.flatten = Flatten(start_axis=1, stop_axis=-1)

        self.fc = Linear(
            int(1024 * scale),
            class_num,
            weight_attr=ParamAttr(initializer=KaimingNormal()))
        
    def forward(self, x):
        x = self.conv(x)
        x = self.blocks(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def MobileNetV1_x0_25(**args):
    """
        MobileNetV1_x0_25
        Args:
            pretrained: bool=False. If `True` load pretrained parameters, `False` otherwise.
            kwargs: 
                class_num: int=1000. Output dim of last fc layer.
        Returns:
            model: nn.Layer. Specific `MobileNetV1_x0_25` model depends on args.
    """
    model = MobileNet(scale=0.25, **args)
    if isinstance(model.pretrained, bool):
        if model.pretrained is True:
            load_dygraph_pretrain_from_url(model, MODEL_URLS["MobileNetV1_x0_25"])
    elif isinstance(model.pretrained, str):
        load_dygraph_pretrain(model, model.pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type")
    return model


def MobileNetV1_x0_5(**args):
    """
        MobileNetV1_x0_5
        Args:
            pretrained: bool=False. If `True` load pretrained parameters, `False` otherwise.
            kwargs: 
                class_num: int=1000. Output dim of last fc layer.
        Returns:
            model: nn.Layer. Specific `MobileNetV1_x0_5` model depends on args.
    """
    model = MobileNet(scale=0.5, **args)
    if isinstance(model.pretrained, bool):
        if model.pretrained is True:
            load_dygraph_pretrain_from_url(model, MODEL_URLS["MobileNetV1_x0_5"])
    elif isinstance(model.pretrained, str):
        load_dygraph_pretrain(model, model.pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type")
    return model


def MobileNetV1_x0_75(**args):
    """
        MobileNetV1_x0_75
        Args:
            pretrained: bool=False. If `True` load pretrained parameters, `False` otherwise.
            kwargs: 
                class_num: int=1000. Output dim of last fc layer.
        Returns:
            model: nn.Layer. Specific `MobileNetV1_x0_75` model depends on args.
    """
    model = MobileNet(scale=0.75, **args)
    if isinstance(model.pretrained, bool):
        if model.pretrained is True:
            load_dygraph_pretrain_from_url(model, MODEL_URLS["MobileNetV1_x0_75"])
    elif isinstance(model.pretrained, str):
        load_dygraph_pretrain(model, model.pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type")
    return model


def MobileNetV1(**args):
    """
        MobileNetV1
        Args:
            pretrained: bool=False. If `True` load pretrained parameters, `False` otherwise.
            kwargs: 
                class_num: int=1000. Output dim of last fc layer.
        Returns:
            model: nn.Layer. Specific `MobileNetV1` model depends on args.
    """
    model = MobileNet(scale=1.0, **args)
    if isinstance(model.pretrained, bool):
        if model.pretrained is True:
            load_dygraph_pretrain_from_url(model, MODEL_URLS["MobileNetV1"])
    elif isinstance(model.pretrained, str):
        load_dygraph_pretrain(model, model.pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type")
    return model


