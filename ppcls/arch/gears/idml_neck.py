# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import math

import paddle
import paddle.nn as nn
from ppcls.utils.initializer import kaiming_normal_


# This neck is just for reproduction of  paper(Introspective Deep Metric Learning)
class IDMLNeck(nn.Layer):
    def __init__(self, in_channel_num, embedding_size, is_norm=True, bias=True):
        super().__init__()
        self.in_channel_num = in_channel_num
        self.embedding_size = embedding_size
        self.is_norm = is_norm
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.gmp = nn.AdaptiveMaxPool2D(1)
        
        kernel_weight = paddle.empty([self.in_channel_num, self.embedding_size])
        kernel_weight = kaiming_normal_(kernel_weight, mode='fan_out')
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(kernel_weight))
        self.embedding_layer = nn.Linear(self.in_channel_num, self.embedding_size, bias_attr=bias, weight_attr=weight_attr)
        self.uncertainty_layer = nn.Linear(self.in_channel_num, self.embedding_size, bias_attr=bias, weight_attr=weight_attr)
        
    def l2_norm(self, input, axis=1):
        norm = paddle.sqrt(
            paddle.sum(paddle.square(input), axis=axis, keepdim=True).add(paddle.to_tensor(1e-12))) 
        output = paddle.divide(input, norm)
        return output
    
    def forward(self, x):
        avg_x = self.gap(x)
        max_x = self.gmp(x)
        x = max_x + avg_x
        x = x.reshape([x.shape[0], -1])
        x_semantic = self.embedding_layer(x)
        if self.training:
            x_uncertainty = self.uncertainty_layer(x)
        if self.is_norm:
            x_semantic = self.l2_norm(x_semantic, axis=1)
            if self.training:
                x_uncertainty = self.l2_norm(x_uncertainty, axis=1)
        if self.training:
            return paddle.concat([x_semantic, x_uncertainty], axis=0)
        else:
            return x_semantic