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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import numpy as np
import paddle.fluid as fluid


def instance_std(input, epsilon=1e-5):
    v = paddle.var(input, axis=[2, 3], keepdim=True)
    v = paddle.expand_as(v, input)
    return paddle.sqrt(v + epsilon)


def group_std(input, groups=32, epsilon=1e-5):
    N, C, H, W = input.shape
    print(N, C, H, W)
    input = paddle.reshape(input, [N, groups, C // groups, H, W])
    v = paddle.var(input, axis=[2, 3, 4], keepdim=True)
    v = paddle.expand_as(v, input)
    return paddle.reshape(paddle.sqrt(v + epsilon), (N, C, H, W))


class EvoNorm(fluid.dygraph.Layer):
    def __init__(self,
                 channels,
                 version='S0',
                 non_linear=True,
                 groups,
                 epsilon,
                 momentum,
                 training):
        super(EvoNorm, self).__init__()
        self.channels = channels
        self.version = version
        self.non_linear = non_linear
        self.groups = groups
        self.epsilon = epsilon
        self.training = training
        self.momentum = momentum

        self.gamma = self.create_parameter(
            [1, self.channels, 1, 1],
            default_initializer=fluid.initializer.Constant(value=0.0))
        self.beta = self.create_parameter(
            [1, self.channels, 1, 1],
            default_initializer=fluid.initializer.Constant(value=0.0))
        if self.non_linear:
            self.v = self.create_parameter(
                [1, self.channels, 1, 1],
                default_initializer=fluid.initializer.Constant(value=0.0))
        self.running_var = self.create_parameter(
            [1, self.channels, 1, 1],
            default_initializer=fluid.initializer.Constant(value=0.0))
        self.running_var.stop_gradient = True

    def forward(self, input):
        if self.version == 'S0':
            if self.non_linear:
                num = input * paddle.sigmoid(self.v * input)
                return num / group_std(
                    input, groups=self.groups,
                    epsilon=self.epsilon) * self.gamma + self.beta
            else:
                return input * self.gamma + self.beta

        if self.version == 'B0':
            if self.training:
                var = paddle.var(input,
                                 axis=[0, 2, 3],
                                 unbiased=False,
                                 keepdim=True)
                self.running_var = self.running_var * self.momentum
                self.running_var = self.running_var + (1 - self.momentum) * var
            else:
                var = self.running_var
            if self.non_linear:
                den = paddle.elementwise_max(
                    paddle.sqrt((var + self.epsilon)),
                    self.v * x + instance_std(x))
                return x / den * self.gamma + self.beta
            else:
                return x * self.gamma + self.beta
