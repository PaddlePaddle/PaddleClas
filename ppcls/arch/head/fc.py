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

import paddle
import paddle.nn as nn
from paddle import Tensor


class FC(nn.Layer):
    def __init__(self, embedding_size: int, class_num: int):
        super(FC, self).__init__()
        self.embedding_size = embedding_size
        self.class_num = class_num
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierNormal())
        self.fc = paddle.nn.Linear(
            self.embedding_size, self.class_num, weight_attr=weight_attr)

    def forward(self, input: Tensor) -> Tensor:
        out = self.fc(input)
        return out
