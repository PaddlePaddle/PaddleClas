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

import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils import logger


class BCELoss(nn.Layer):
    """
    Binary Cross entropy loss
    """

    def __init__(self,
                 epsilon=0.0,
                 target_threshold=0.2,
                 weight=None,
                 reduction="mean",
                 pos_weight=None):
        super().__init__()
        assert 0. <= epsilon < 1.0
        self.epsilon = epsilon
        self.target_threshold = target_threshold
        self.reduction = reduction
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label):
        if isinstance(x, dict):
            x = x["logits"]
        class_num = x.shape[-1]
        target = self._labelsmoothing(label, class_num)
        target = target.__gt__(self.target_threshold).astype("float32")
        loss = F.binary_cross_entropy_with_logits(
            x,
            target,
            self.weight,
            pos_weight=self.pos_weight,
            reduction=self.reduction)
        loss = loss.mean()
        return {"BCELoss": loss}
