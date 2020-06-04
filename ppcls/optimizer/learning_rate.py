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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math

import paddle.fluid as fluid
import paddle.fluid.layers.ops as ops
from paddle.fluid.layers.learning_rate_scheduler import _decay_step_counter

__all__ = ['LearningRateBuilder']


class Linear(object):
    """
    Linear learning rate decay

    Args:
        lr(float): initial learning rate
        steps(int): total decay steps
        end_lr(float): end learning rate, default: 0.0.
    """

    def __init__(self, lr, steps, end_lr=0.0, **kwargs):
        super(Linear, self).__init__()
        self.lr = lr
        self.steps = steps
        self.end_lr = end_lr

    def __call__(self):
        learning_rate = fluid.layers.polynomial_decay(
            self.lr, self.steps, self.end_lr, power=1)
        return learning_rate


class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
    """

    def __init__(self, lr, step_each_epoch, epochs, **kwargs):
        super(Cosine, self).__init__()
        self.lr = lr
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs

    def __call__(self):
        learning_rate = fluid.layers.cosine_decay(
            learning_rate=self.lr,
            step_each_epoch=self.step_each_epoch,
            epochs=self.epochs)
        return learning_rate


class Piecewise(object):
    """
    Piecewise learning rate decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(list): piecewise decay epochs
        gamma(float): decay factor
    """

    def __init__(self, lr, step_each_epoch, decay_epochs, gamma=0.1, **kwargs):
        super(Piecewise, self).__init__()
        self.bd = [step_each_epoch * e for e in decay_epochs]
        self.lr = [lr * (gamma**i) for i in range(len(self.bd) + 1)]

    def __call__(self):
        learning_rate = fluid.layers.piecewise_decay(self.bd, self.lr)
        return learning_rate


class CosineWarmup(object):
    """
    Cosine learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): cosine decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self, lr, step_each_epoch, epochs, warmup_epoch=5, **kwargs):
        super(CosineWarmup, self).__init__()
        self.lr = lr
        self.step_each_epoch = step_each_epoch
        self.epochs = epochs
        self.warmup_epoch = fluid.layers.fill_constant(
            shape=[1],
            value=float(warmup_epoch),
            dtype='float32',
            force_cpu=True)

    def __call__(self):
        global_step = _decay_step_counter()
        learning_rate = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")
        epoch = ops.floor(global_step / self.step_each_epoch)
        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(epoch < self.warmup_epoch):
                decayed_lr = self.lr * \
                    (global_step / (self.step_each_epoch * self.warmup_epoch))
                fluid.layers.tensor.assign(
                    input=decayed_lr, output=learning_rate)
            with switch.default():
                current_step = global_step - self.warmup_epoch * self.step_each_epoch
                total_step = (
                    self.epochs - self.warmup_epoch) * self.step_each_epoch
                decayed_lr = self.lr * \
                    (ops.cos(current_step * math.pi / total_step) + 1) / 2
                fluid.layers.tensor.assign(
                    input=decayed_lr, output=learning_rate)

        return learning_rate


class ExponentialWarmup(object):
    """
    Exponential learning rate decay with warmup
    [0, warmup_epoch): linear warmup
    [warmup_epoch, epochs): Exponential decay

    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        decay_epochs(float): decay epochs
        decay_rate(float): decay rate
        warmup_epoch(int): epoch num of warmup
    """

    def __init__(self,
                 lr,
                 step_each_epoch,
                 decay_epochs=2.4,
                 decay_rate=0.97,
                 warmup_epoch=5,
                 **kwargs):
        super(ExponentialWarmup, self).__init__()
        self.lr = lr
        self.step_each_epoch = step_each_epoch
        self.decay_epochs = decay_epochs * self.step_each_epoch
        self.decay_rate = decay_rate
        self.warmup_epoch = fluid.layers.fill_constant(
            shape=[1],
            value=float(warmup_epoch),
            dtype='float32',
            force_cpu=True)

    def __call__(self):
        global_step = _decay_step_counter()
        learning_rate = fluid.layers.tensor.create_global_var(
            shape=[1],
            value=0.0,
            dtype='float32',
            persistable=True,
            name="learning_rate")

        epoch = ops.floor(global_step / self.step_each_epoch)
        with fluid.layers.control_flow.Switch() as switch:
            with switch.case(epoch < self.warmup_epoch):
                decayed_lr = self.lr * \
                    (global_step / (self.step_each_epoch * self.warmup_epoch))
                fluid.layers.tensor.assign(
                    input=decayed_lr, output=learning_rate)
            with switch.default():
                rest_step = global_step - self.warmup_epoch * self.step_each_epoch
                div_res = ops.floor(rest_step / self.decay_epochs)

                decayed_lr = self.lr * (self.decay_rate**div_res)
                fluid.layers.tensor.assign(
                    input=decayed_lr, output=learning_rate)

        return learning_rate


class LearningRateBuilder():
    """
    Build learning rate variable
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/layers_cn.html

    Args:
        function(str): class name of learning rate
        params(dict): parameters used for init the class
    """

    def __init__(self,
                 function='Linear',
                 params={'lr': 0.1,
                         'steps': 100,
                         'end_lr': 0.0}):
        self.function = function
        self.params = params

    def __call__(self):
        mod = sys.modules[__name__]
        lr = getattr(mod, self.function)(**self.params)()
        return lr
