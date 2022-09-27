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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import abstractmethod
from typing import Union

from paddle.optimizer import lr
from ppcls.utils import logger


class LRBase(object):
    """Base class for custom learning rates

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        warmup_epoch (int): number of warmup epoch(s)
        warmup_start_lr (float): start learning rate within warmup
        last_epoch (int): last epoch
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter
        verbose (bool): If True, prints a message to stdout for each update. Defaults to False
    """

    def __init__(self,
                 epochs: int,
                 step_each_epoch: int,
                 learning_rate: float,
                 warmup_epoch: int,
                 warmup_start_lr: float,
                 last_epoch: int,
                 by_epoch: bool,
                 verbose: bool=False) -> None:
        """Initialize and record the necessary parameters
        """
        super(LRBase, self).__init__()
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"Global.epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logger.warning(msg)
            warmup_epoch = epochs
        self.epochs = epochs
        self.step_each_epoch = step_each_epoch
        self.learning_rate = learning_rate
        self.warmup_epoch = warmup_epoch
        self.warmup_steps = self.warmup_epoch if by_epoch else round(
            self.warmup_epoch * self.step_each_epoch)
        self.warmup_start_lr = warmup_start_lr
        self.last_epoch = last_epoch
        self.by_epoch = by_epoch
        self.verbose = verbose

    @abstractmethod
    def __call__(self, *kargs, **kwargs) -> lr.LRScheduler:
        """generate an learning rate scheduler

        Returns:
            lr.LinearWarmup: learning rate scheduler
        """
        pass

    def linear_warmup(
            self,
            learning_rate: Union[float, lr.LRScheduler]) -> lr.LinearWarmup:
        """Add an Linear Warmup before learning_rate

        Args:
            learning_rate (Union[float, lr.LRScheduler]): original learning rate without warmup

        Returns:
            lr.LinearWarmup: learning rate scheduler with warmup
        """
        warmup_lr = lr.LinearWarmup(
            learning_rate=learning_rate,
            warmup_steps=self.warmup_steps,
            start_lr=self.warmup_start_lr,
            end_lr=self.learning_rate,
            last_epoch=self.last_epoch,
            verbose=self.verbose)
        return warmup_lr


class Constant(lr.LRScheduler):
    """Constant learning rate Class implementation

    Args:
        learning_rate (float): The initial learning rate
        last_epoch (int, optional): The index of last epoch. Default: -1.
    """

    def __init__(self, learning_rate, last_epoch=-1, **kwargs):
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        super(Constant, self).__init__()

    def get_lr(self) -> float:
        """always return the same learning rate
        """
        return self.learning_rate


class ConstLR(LRBase):
    """Constant learning rate

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        warmup_epoch (int): number of warmup epoch(s)
        warmup_start_lr (float): start learning rate within warmup
        last_epoch (int): last epoch
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(ConstLR, self).__init__(epochs, step_each_epoch, learning_rate,
                                      warmup_epoch, warmup_start_lr,
                                      last_epoch, by_epoch)

    def __call__(self):
        learning_rate = Constant(
            learning_rate=self.learning_rate, last_epoch=self.last_epoch)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Linear(LRBase):
    """Linear learning rate decay

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        end_lr (float, optional): The minimum final learning rate. Defaults to 0.0.
        power (float, optional): Power of polynomial. Defaults to 1.0.
        warmup_epoch (int): number of warmup epoch(s)
        warmup_start_lr (float): start learning rate within warmup
        last_epoch (int): last epoch
        by_epoch (bool): learning rate decays by epoch when by_epoch is True, else by iter
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 end_lr=0.0,
                 power=1.0,
                 cycle=False,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(Linear, self).__init__(epochs, step_each_epoch, learning_rate,
                                     warmup_epoch, warmup_start_lr, last_epoch,
                                     by_epoch)
        self.decay_steps = (epochs - self.warmup_epoch) * step_each_epoch
        self.end_lr = end_lr
        self.power = power
        self.cycle = cycle
        self.warmup_steps = round(self.warmup_epoch * step_each_epoch)
        if self.by_epoch:
            self.decay_steps = self.epochs - self.warmup_epoch

    def __call__(self):
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.decay_steps,
            end_lr=self.end_lr,
            power=self.power,
            cycle=self.cycle,
            last_epoch=self.last_epoch) if self.decay_steps > 0 else Constant(
                self.learning_rate)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Cosine(LRBase):
    """Cosine learning rate decay

    ``lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)``

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        eta_min (float, optional): Minimum learning rate. Defaults to 0.0.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 eta_min=0.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(Cosine, self).__init__(epochs, step_each_epoch, learning_rate,
                                     warmup_epoch, warmup_start_lr, last_epoch,
                                     by_epoch)
        self.T_max = (self.epochs - self.warmup_epoch) * self.step_each_epoch
        self.eta_min = eta_min
        if self.by_epoch:
            self.T_max = self.epochs - self.warmup_epoch

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            eta_min=self.eta_min,
            last_epoch=self.last_epoch) if self.T_max > 0 else Constant(
                self.learning_rate)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Step(LRBase):
    """Step learning rate decay

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma``. It should be less than 1.0. Default: 0.1.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 step_size,
                 gamma,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(Step, self).__init__(epochs, step_each_epoch, learning_rate,
                                   warmup_epoch, warmup_start_lr, last_epoch,
                                   by_epoch)
        self.step_size = step_size * step_each_epoch
        self.gamma = gamma
        if self.by_epoch:
            self.step_size = step_size

    def __call__(self):
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class Piecewise(LRBase):
    """Piecewise learning rate decay

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        decay_epochs (List[int]): A list of steps numbers. The type of element in the list is python int.
        values (List[float]): A list of learning rate values that will be picked during different epoch boundaries.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 decay_epochs,
                 values,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(Piecewise,
              self).__init__(epochs, step_each_epoch, values[0], warmup_epoch,
                             warmup_start_lr, last_epoch, by_epoch)
        self.values = values
        self.boundaries_steps = [e * step_each_epoch for e in decay_epochs]
        if self.by_epoch is True:
            self.boundaries_steps = decay_epochs

    def __call__(self):
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries_steps,
            values=self.values,
            last_epoch=self.last_epoch)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class MultiStepDecay(LRBase):
    """MultiStepDecay learning rate decay

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        milestones (List[int]): List of each boundaries. Must be increasing.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma``. It should be less than 1.0. Defaults to 0.1.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 milestones,
                 gamma=0.1,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(MultiStepDecay, self).__init__(
            epochs, step_each_epoch, learning_rate, warmup_epoch,
            warmup_start_lr, last_epoch, by_epoch)
        self.milestones = [x * step_each_epoch for x in milestones]
        self.gamma = gamma
        if self.by_epoch:
            self.milestones = milestones

    def __call__(self):
        learning_rate = lr.MultiStepDecay(
            learning_rate=self.learning_rate,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate
