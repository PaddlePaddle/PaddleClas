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


# This neck is just for reproduction of  paper(Introspective Deep Metric Learning)
class IDMLNeck(nn.Layer):
    def __init__(self, in_channel_num, embedding_size, is_norm=True, bias=True):
        super().__init__()
        self.in_channel_num = in_channel_num
        self.embedding_size = embedding_size
        self.is_norm = is_norm
        self.gap = nn.AdaptiveAvgPool2D(1)
        self.gmp = nn.AdaptiveMaxPool2D(1)
        
        kernel_weight = paddle.uniform(
            [ self.in_channel_num, self.embedding_size], min=-1, max=1)
        
        kernel_weight_norm = paddle.norm(
            kernel_weight, p=2, axis=0, keepdim=True)
        kernel_weight_norm = paddle.where(kernel_weight_norm > 1e-5,
                                          kernel_weight_norm,
                                          paddle.ones_like(kernel_weight_norm))
        kernel_weight = kernel_weight / kernel_weight_norm
        
        weight_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(kernel_weight))
        self.embedding_layer = nn.Linear(self.in_channel_num, self.embedding_size, bias_attr=bias, weight_attr=weight_attr)
        self.uncertainty_layer = nn.Linear(self.in_channel_num, self.embedding_size, bias_attr=bias)
        
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


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out

def kaiming_normal_(tensor, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    def _calculate_correct_fan(tensor, mode):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError(
                "Mode {} not supported, please use one of {}".format(
                    mode, valid_modes))

        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == 'fan_in' else fan_out

    def calculate_gain(nonlinearity, param=None):
        linear_fns = [
            'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
            'conv_transpose2d', 'conv_transpose3d'
        ]
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1
        elif nonlinearity == 'tanh':
            return 5.0 / 3
        elif nonlinearity == 'relu':
            return math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(
                    param, int) or isinstance(param, float):
                negative_slope = param
            else:
                raise ValueError(
                    "negative_slope {} not a valid number".format(param))
            return math.sqrt(2.0 / (1 + negative_slope**2))
        else:
            raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    with paddle.no_grad():
        paddle.nn.initializer.Normal(0, std)(tensor)
        return tensor
