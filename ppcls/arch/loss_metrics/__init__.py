#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


# TODO: fix the format
class CELoss(nn.Layer):
    """
    """

    def __init__(self, name="loss", epsilon=None):
        super().__init__()
        self.name = name
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon

    def _labelsmoothing(self, target, class_num):
        if target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, label, mode="train"):
        loss_dict = {}
        if self.epsilon is not None:
            class_num = x.shape[-1]
            label = self._labelsmoothing(label, class_num)
            x = -F.log_softmax(x, axis=-1)
            loss = paddle.sum(x * label, axis=-1)
        else:
            if label.shape[-1] == x.shape[-1]:
                label = F.softmax(label, axis=-1)
                soft_label = True
            else:
                soft_label = False
            loss = F.cross_entropy(x, label=label, soft_label=soft_label)
        loss_dict[self.name] = paddle.mean(loss)
        return loss_dict


# TODO: fix the format
class Topk(nn.Layer):
    def __init__(self, topk=[1, 5]):
        super().__init__()
        assert isinstance(topk, (int, list))
        if isinstance(topk, int):
            topk = [topk]
        self.topk = topk

    def forward(self, x, label):
        metric_dict = dict()
        for k in self.topk:
            metric_dict["top{}".format(k)] = paddle.metric.accuracy(
                x, label, k=k)
        return metric_dict


# TODO: fix the format
def build_loss(config):
    config = list(copy.deepcopy(config[0]).values())[0]
    weight = config.pop("weight")
    loss_func = CELoss(**config)
    return loss_func


# TODO: fix the format
def build_metrics(config):
    metrics_func = Topk()
    return metrics_func
