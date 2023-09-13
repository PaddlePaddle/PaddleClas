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

from ppcls.loss.multilabelloss import ratio2weight_1, ratio2weight_2, ratio2weight_3


class DMLLoss(nn.Layer):
    """
    DMLLoss
    """

    def __init__(self, act="softmax", sum_across_class_dim=False, eps=1e-12, weight_type=1, weight_alpha=0.1):
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
        self.sum_across_class_dim = sum_across_class_dim
        self.weight_type = weight_type
        self.weight_alpha = weight_alpha

    def _kldiv(self, x, target):
        class_num = x.shape[-1]
        cost = target * paddle.log(
            (target + self.eps) / (x + self.eps)) * class_num
        return cost

    def forward(self, x, target, gt_label=None):
        if self.act is not None:
            x = self.act(x)
            target = self.act(target)
        loss = self._kldiv(x, target) + self._kldiv(target, x)
        loss = loss / 2

        # for multi-label dml loss
        if gt_label is not None:
            gt_label, label_ratio = gt_label[:, 0, :], gt_label[:, 1, :]
            targets_mask = paddle.cast(gt_label > 0.5, 'float32')
            if self.weight_type == 2:
                weight = ratio2weight_2(
                    targets_mask, paddle.to_tensor(label_ratio))
            elif self.weight_type == 3:
                weight = ratio2weight_3(
                    targets_mask, paddle.to_tensor(label_ratio), self.weight_alpha)
            else:
                weight = ratio2weight_1(
                    targets_mask, paddle.to_tensor(label_ratio))
            weight = weight * (gt_label > -1)
            loss = loss * weight

        loss = loss.sum(1).mean() if self.sum_across_class_dim else loss.mean()
        return {"DMLLoss": loss}
