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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle
import paddle.nn as nn
from paddle import Tensor
from ppcls.arch.utils import _calculate_fan_in_and_fan_out, kaiming_uniform_


class CenterLossHead(nn.Layer):
    def __init__(self, input_dims: int, class_num: int):
        super(CenterLossHead, self).__init__()
        self.input_dims = input_dims
        self.class_num = class_num
        self.fc = paddle.nn.Linear(self.input_dims, self.class_num)

        kaiming_uniform_(self.fc.weight, a=math.sqrt(5))
        if self.fc.bias is not None:
            fan_in, _ = _calculate_fan_in_and_fan_out(self.fc.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.initializer.Uniform(-bound, bound)(self.fc.bias)

    def forward(self, input: Tensor) -> Tensor:
        out = self.fc(input)
        return out
