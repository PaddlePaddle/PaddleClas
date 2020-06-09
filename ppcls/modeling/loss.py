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

import paddle.fluid as fluid

__all__ = ['CELoss', 'MixCELoss', 'GoogLeNetLoss', 'JSDivLoss']


class Loss(object):
    """
    Loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True
        else:
            self._epsilon = None
            self._label_smoothing = False

    def _labelsmoothing(self, target):
        if target.shape[-1] != self._class_dim:
            one_hot_target = fluid.one_hot(input=target, depth=self._class_dim)
        else:
            one_hot_target = target
        soft_target = fluid.layers.label_smooth(
            label=one_hot_target, epsilon=self._epsilon, dtype="float32")
        soft_target = fluid.layers.reshape(
            soft_target, shape=[-1, self._class_dim])
        soft_target.stop_gradient = True
        return soft_target

    def _crossentropy(self, input, target):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
        softmax_out = fluid.layers.softmax(input, use_cudnn=False)
        cost = fluid.layers.cross_entropy(
            input=softmax_out, label=target, soft_label=self._label_smoothing)
        avg_cost = fluid.layers.mean(cost)
        return avg_cost

    def _kldiv(self, input, target):
        cost = target * fluid.layers.log(target / input) * self._class_dim
        cost = fluid.layers.sum(cost)
        return cost

    def _jsdiv(self, input, target):
        input = fluid.layers.softmax(input, use_cudnn=False)
        target = fluid.layers.softmax(target, use_cudnn=False)
        cost = self._kldiv(input, target) + self._kldiv(target, input)
        cost = cost / 2
        avg_cost = fluid.layers.mean(cost)
        return avg_cost

    def __call__(self, input, target):
        pass


class CELoss(Loss):
    """
    Cross entropy loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(CELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._crossentropy(input, target)
        return cost


class MixCELoss(Loss):
    """
    Cross entropy loss with mix(mixup, cutmix, fixmix)
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(MixCELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target0, target1, lam):
        cost0 = self._crossentropy(input, target0)
        cost1 = self._crossentropy(input, target1)
        cost = lam * cost0 + (1.0 - lam) * cost1
        avg_cost = fluid.layers.mean(cost)
        return avg_cost


class GoogLeNetLoss(Loss):
    """
    Cross entropy loss used after googlenet
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(GoogLeNetLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input0, input1, input2, target):
        cost0 = self._crossentropy(input0, target)
        cost1 = self._crossentropy(input1, target)
        cost2 = self._crossentropy(input2, target)
        cost = cost0 + 0.3 * cost1 + 0.3 * cost2
        avg_cost = fluid.layers.mean(cost)
        return avg_cost


class JSDivLoss(Loss):
    """
    JSDiv loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(JSDivLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._jsdiv(input, target)
        return cost
