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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, act="softmax", eps=1e-12):
        super().__init__()
        if act is not None:
            assert act in ["softmax", "sigmoid"]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.eps = eps

    def _kldiv(self, x, target):
        class_num = x.shape[-1]
        cost = target * paddle.log(
            (target + self.eps) / (x + self.eps)) * class_num
        return cost

    def forward(self, x, target):
        if self.act is not None:
            x = self.act(x)
            target = self.act(target)
        loss = self._kldiv(x, target) + self._kldiv(target, x)
        loss = loss / 2
        loss = paddle.mean(loss)
        return {"DMLLoss": loss}
