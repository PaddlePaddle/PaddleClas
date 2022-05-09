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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn

from ppcls.arch.utils import get_param_attr_dict


class BNNeck(nn.Layer):
    def __init__(self, num_features, **kwargs):
        super().__init__()
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0),
            trainable=False)

        if 'weight_attr' in kwargs:
            weight_attr = get_param_attr_dict(kwargs['weight_attr'])

        bias_attr = None
        if 'bias_attr' in kwargs:
            bias_attr = get_param_attr_dict(kwargs['bias_attr'])

        self.feat_bn = nn.BatchNorm1D(
            num_features,
            momentum=0.9,
            epsilon=1e-05,
            weight_attr=weight_attr,
            bias_attr=bias_attr)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.flatten(x)
        x = self.feat_bn(x)
        return x
