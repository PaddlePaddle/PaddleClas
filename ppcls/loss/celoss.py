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

import os
import warnings

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils import logger


class CELoss(nn.Layer):
    """
    Cross entropy loss
    """

    def __init__(self, reduction="mean", epsilon=None, weight_file=None):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction
        self.weight_data = self._read_weight(
            weight_file) if weight_file else None

    def _read_weight(self, weight_file):
        if not os.path.exists(weight_file):
            msg = f"The file of rescaling weight is not exists. And the setting has been ignored. Please check the file path: {weight_file}."
            logger.warning(msg)
            return None
        else:
            with open(weight_file, "r") as f:
                lines = f.readlines()

            try:
                weight_list = []
                for line in lines:
                    weight_list.append(float(line.strip()))
            except Exception as e:
                msg = ""
                logger.warning(msg)
                return None
            return paddle.to_tensor(weight_list)

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
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            soft_label = True
        else:
            if label.shape[-1] == x.shape[-1]:
                soft_label = True
            else:
                soft_label = False

        if self.weight_data is not None:
            if self.weight_data.shape[0] != class_num:
                msg = f"The shape of rescaling weight must be [class num]. Please check the rescaling weight file. The setting has been ignored."
                logger.warning(msg)
                self.weight_data = None

        loss = F.cross_entropy(
            x, label=label, soft_label=soft_label, reduction=self.reduction, weight=self.weight_data)
        return {"CELoss": loss}


class MixCELoss(object):
    def __init__(self, *args, **kwargs):
        msg = "\"MixCELoss\" is deprecated, please use \"CELoss\" instead."
        logger.error(DeprecationWarning(msg))
        raise DeprecationWarning(msg)
