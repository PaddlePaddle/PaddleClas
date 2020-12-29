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

import sys
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class CELoss(nn.Layer):
    '''
    Cross entropy loss
    Args:
    '''

    def __init__(self, classes_num=1000, epsilon=None, mode="mean", **args):
        super(CELoss, self).__init__()
        self._mode = mode
        assert mode in [
            "mean", "sum"
        ], "mode must be in [mean, sum], but got {]".format(mode)
        self._classes_num = classes_num
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    def reduce_loss(self, cost):
        if self._mode == "mean":
            avg_cost = paddle.mean(cost)
        else:
            avg_cost = paddle.sum(cost)
        return avg_cost

    def _labelsmoothing(self, target):
        if target.shape[-1] != self._classes_num:
            one_hot_target = F.one_hot(target, self._classes_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)
        soft_target = paddle.reshape(
            soft_target, shape=[-1, self._classes_num])
        return soft_target

    def calc_loss(self, x, target):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            x = -F.log_softmax(x, axis=-1)
            cost = paddle.sum(x * target, axis=-1)
        else:
            cost = F.cross_entropy(x, label=target)
        avg_cost = self.reduce_loss(cost)
        return avg_cost

    def forward(self, x, feeds):
        target = feeds["label"]
        return self.calc_loss(x, target)


class MixCELoss(CELoss):
    '''
    Mix cross entropy loss
    Args:
    '''

    def __init__(self, classes_num=1000, epsilon=None, mode="mean", **args):
        super(MixCELoss, self).__init__(
            classes_num=classes_num, epsilon=epsilon, mode=mode, **args)

    def forward(self, x, feeds):
        target0 = feeds['y_a']
        target1 = feeds['y_b']
        lam = feeds['lam']
        cost0 = self.calc_loss(x, target0)
        cost1 = self.calc_loss(x, target1)
        cost = lam * cost0 + (1.0 - lam) * cost1
        avg_cost = self.reduce_loss(cost)
        return avg_cost


class JSDivLoss(nn.Layer):
    '''
    JSDiv loss
    Args:
    '''

    def __init__(self, classes_num=1000):
        super(JSDivLoss, self).__init__()
        self._classes_num = classes_num

    def _kldiv(self, x, target, eps=1e-10):
        cost = target * paddle.log(
            (target + eps) / (x + eps)) * self._classes_num
        return cost

    def forward(self, x, target):
        x = F.softmax(x)
        target = F.softmax(target)
        cost = self._kldiv(x, target) + self._kldiv(target, x)
        cost = cost / 2
        avg_cost = paddle.mean(cost)
        return avg_cost


class GoogLeNetLoss(CELoss):
    '''
    GoogLeNet loss
    Args:
    '''

    def __init__(self, classes_num=1000, epsilon=None, mode="mean", **args):
        super(GoogLeNetLoss, self).__init__(
            classes_num=classes_num, epsilon=epsilon, mode=mode, **args)

    def forward(self, x, feeds):
        assert len(
            x
        ) == 3, "input for {} must be 3 but got {}, please check your input".format(
        )
        target = feeds["label"]
        cost0 = self.calc_loss(x[0], target)
        cost1 = self.calc_loss(x[1], target)
        cost2 = self.calc_loss(x[2], target)
        cost = cost0 + 0.3 * cost1 + 0.3 * cost2

        avg_cost = self.reduce_loss(cost)
        return avg_cost


class LossBuilder(object):
    """
    Build loss

    Args:
        function(str): loss name of learning rate
        params(dict): parameters used for init the class
    """

    def __init__(self, function='CELoss', params={}):
        self.function = function
        self.params = params

    def __call__(self):
        mod = sys.modules[__name__]
        func = getattr(mod, self.function)
        return func(**self.params)


def test_ce_loss():
    x = paddle.randn((16, 1000))
    label = paddle.ones((16, 1), dtype="int64")
    for eps in [0.0, 0.1, 1.0]:
        celoss_func = CrossEntropyLoss(
            classes_num=1000, epsilon=eps, mode="mean")
        loss = celoss_func(x, label)
        print(loss)


def test_jsdiv_loss():
    x = paddle.randn((16, 1000))
    y = paddle.randn((16, 1000))
    loss_func = JSDivLoss(classes_num=1000)
    loss = loss_func(x, y)
    print(loss)


if __name__ == "__main__":
    test_ce_loss()
    test_jsdiv_loss()
