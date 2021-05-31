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

import paddle
import paddle.nn.functional as F

__all__ = ['CELoss', 'JSDivLoss', 'KLDivLoss']


class Loss(object):
    """
    Loss
    """
    def __init__(self, class_dim=1000, epsilon=None):
        assert class_dim > 1, "class_dim=%d is not larger than 1" % (class_dim)
        self._class_dim = class_dim
        if epsilon is not None and epsilon >= 0.0 and epsilon <= 1.0:
            self._epsilon = epsilon
            self._label_smoothing = True  #use label smoothing.(Actually, it is softmax label)
        else:
            self._epsilon = None
            self._label_smoothing = False

    #do label_smoothing
    def _labelsmoothing(self, target):
        if target.shape[-1] != self._class_dim:
            one_hot_target = F.one_hot(target, self._class_dim)  #do ont hot(23,34,46)-> 3 * _class_dim
        else:
            one_hot_target = target

        #do label_smooth
        soft_target = F.label_smooth(one_hot_target, epsilon=self._epsilon)   #(1 - epsilon) * input + eposilon / K.
        soft_target = paddle.reshape(soft_target, shape=[-1, self._class_dim])
        return soft_target

    def _crossentropy(self, input, target, use_pure_fp16=False):
        if self._label_smoothing:
            target = self._labelsmoothing(target)
            input = -F.log_softmax(input, axis=-1)      #softmax and do log
            cost = paddle.sum(target * input, axis=-1)  #sum  
        else:
            cost = F.cross_entropy(input=input, label=target) 

        if use_pure_fp16:
            avg_cost = paddle.sum(cost)
        else:
            avg_cost = paddle.mean(cost)
        return avg_cost

    def _kldiv(self, input, target, name=None):
        eps = 1.0e-10
        cost = target * paddle.log(
            (target + eps) / (input + eps)) * self._class_dim
        return cost

    def _jsdiv(self, input, target):  #so the input and target is the fc output; no softmax
        input = F.softmax(input)
        target = F.softmax(target) 

        #two distribution
        cost = self._kldiv(input, target) + self._kldiv(target, input)
        cost = cost / 2
        avg_cost = paddle.mean(cost)
        return avg_cost

    def __call__(self, input, target):
        pass


class CELoss(Loss):
    """
    Cross entropy loss
    """

    def __init__(self, class_dim=1000, epsilon=None):
        super(CELoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target, use_pure_fp16=False):
        logits = input["logits"]
        cost = self._crossentropy(logits, target, use_pure_fp16)
        return {"CELoss": cost}

class JSDivLoss(Loss):
    """
    JSDiv loss
    """
    def __init__(self, class_dim=1000, epsilon=None):
        super(JSDivLoss, self).__init__(class_dim, epsilon)

    def __call__(self, input, target):
        cost = self._jsdiv(input, target)
        return cost


class KLDivLoss(paddle.nn.Layer):
    def __init__(self):
        super(KLDivLoss, self).__init__()

    def __call__(self, p, q, is_logit=True):
        if is_logit:
            p = paddle.nn.functional.softmax(p)
            q = paddle.nn.functional.softmax(q)
        return -(p * paddle.log(q + 1e-8)).sum(1).mean()