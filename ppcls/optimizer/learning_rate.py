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
import math
import types
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


class Cyclic(LRBase):
    """Cyclic learning rate decay

    Args:
        epochs (int): Total epoch(s).
        step_each_epoch (int): Number of iterations within an epoch.
        base_learning_rate (float): Initial learning rate, which is the lower boundary in the cycle. The paper recommends
            that set the base_learning_rate to 1/3 or 1/4 of max_learning_rate.
        max_learning_rate (float): Maximum learning rate in the cycle. It defines the cycle amplitude as above.
            Since there is some scaling operation during process of learning rate adjustment,
            max_learning_rate may not actually be reached.
        warmup_epoch (int): Number of warmup epoch(s).
        warmup_start_lr (float): Start learning rate within warmup.
        step_size_up (int): Number of training steps, which is used to increase learning rate in a cycle.
            The step size of one cycle will be defined by step_size_up + step_size_down. According to the paper, step
            size should be set as at least 3 or 4 times steps in one epoch.
        step_size_down (int, optional): Number of training steps, which is used to decrease learning rate in a cycle.
            If not specified, it's value will initialize to `` step_size_up `` . Default: None.
        mode (str, optional): One of 'triangular', 'triangular2' or 'exp_range'.
            If scale_fn is specified, this argument will be ignored. Default: 'triangular'.
        exp_gamma (float): Constant in 'exp_range' scaling function: exp_gamma**iterations. Used only when mode = 'exp_range'. Default: 1.0.
        scale_fn (function, optional): A custom scaling function, which is used to replace three build-in methods.
            It should only have one argument. For all x >= 0, 0 <= scale_fn(x) <= 1.
            If specified, then 'mode' will be ignored. Default: None.
        scale_mode (str, optional): One of 'cycle' or 'iterations'. Defines whether scale_fn is evaluated on cycle
            number or cycle iterations (total iterations since start of training). Default: 'cycle'.
        last_epoch (int, optional): The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch (bool): Learning rate decays by epoch when by_epoch is True, else by iter.
        verbose: (bool, optional): If True, prints a message to stdout for each update. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 base_learning_rate,
                 max_learning_rate,
                 warmup_epoch,
                 warmup_start_lr,
                 step_size_up,
                 step_size_down=None,
                 mode='triangular',
                 exp_gamma=1.0,
                 scale_fn=None,
                 scale_mode='cycle',
                 by_epoch=False,
                 last_epoch=-1,
                 verbose=False):
        super(Cyclic, self).__init__(
            epochs, step_each_epoch, base_learning_rate, warmup_epoch,
            warmup_start_lr, last_epoch, by_epoch, verbose)
        self.base_learning_rate = base_learning_rate
        self.max_learning_rate = max_learning_rate
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.exp_gamma = exp_gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode

    def __call__(self):
        learning_rate = lr.CyclicLR(
            base_learning_rate=self.base_learning_rate,
            max_learning_rate=self.max_learning_rate,
            step_size_up=self.step_size_up,
            step_size_down=self.step_size_down,
            mode=self.mode,
            exp_gamma=self.exp_gamma,
            scale_fn=self.scale_fn,
            scale_mode=self.scale_mode,
            last_epoch=self.last_epoch,
            verbose=self.verbose)

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
        step_size (int|float): the interval to update.
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
        self.step_size = int(step_size * step_each_epoch)
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
                 learning_rate=None,
                 **kwargs):
        if learning_rate:
            decay_epochs = list(range(0, epochs, 30))
            values = [
                learning_rate * (0.1**i) for i in range(len(decay_epochs))
            ]
            # when total epochs < 30, decay_epochs and values should be
            # [] and [lr] respectively, but paddle dont support.
            if len(decay_epochs) == 1:
                decay_epochs = [epochs]
                values = [values[0], values[0]]
            else:
                decay_epochs = decay_epochs[1:]
            logger.warning(
                "When 'learning_rate' of Piecewise has beed set, "
                "the learning rate scheduler would be set by the rule that lr decay 10 times every 30 epochs. "
                f"So, the 'decay_epochs' and 'values' have been set to {decay_epochs} and {values} respectively."
            )
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


class ReduceOnPlateau(LRBase):
    """ReduceOnPlateau learning rate decay
    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        mode (str, optional): ``'min'`` or ``'max'`` can be selected. Normally, it is ``'min'`` , which means that the
            learning rate will reduce when ``loss`` stops descending. Specially, if it's set to ``'max'``, the learning
            rate will reduce when ``loss`` stops ascending. Defaults to ``'min'``.
        factor (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * factor`` .
            It should be less than 1.0. Defaults to 0.1.
        patience (int, optional): When ``loss`` doesn't improve for this number of epochs, learing rate will be reduced.
            Defaults to 10.
        threshold (float, optional): ``threshold`` and ``threshold_mode`` will determine the minimum change of ``loss`` .
            This make tiny changes of ``loss`` will be ignored. Defaults to 1e-4.
        threshold_mode (str, optional): ``'rel'`` or ``'abs'`` can be selected. In ``'rel'`` mode, the minimum change of ``loss``
            is ``last_loss * threshold`` , where ``last_loss`` is ``loss`` in last epoch. In ``'abs'`` mode, the minimum
            change of ``loss`` is ``threshold`` . Defaults to ``'rel'`` .
        cooldown (int, optional): The number of epochs to wait before resuming normal operation. Defaults to 0.
        min_lr (float, optional): The lower bound of the learning rate after reduction. Defaults to 0.
        epsilon (float, optional): Minimal decay applied to lr. If the difference between new and old lr is smaller than epsilon,
            the update is ignored. Defaults to 1e-8.
        warmup_epoch (int, optional): The epoch numbers for LinearWarmup. Defaults to 0.
        warmup_start_lr (float, optional): start learning rate within warmup. Defaults to 0.0.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 mode='min',
                 factor=0.1,
                 patience=10,
                 threshold=1e-4,
                 threshold_mode='rel',
                 cooldown=0,
                 min_lr=0,
                 epsilon=1e-8,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=False,
                 **kwargs):
        super(ReduceOnPlateau, self).__init__(
            epochs, step_each_epoch, learning_rate, warmup_epoch,
            warmup_start_lr, last_epoch, by_epoch)
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.cooldown = cooldown
        self.min_lr = min_lr
        self.epsilon = epsilon

    def __call__(self):
        learning_rate = lr.ReduceOnPlateau(
            learning_rate=self.learning_rate,
            mode=self.mode,
            factor=self.factor,
            patience=self.patience,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            cooldown=self.cooldown,
            min_lr=self.min_lr,
            epsilon=self.epsilon)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        # NOTE: Implement get_lr() method for class `ReduceOnPlateau`,
        # which is called in `log_info` function
        def get_lr(self):
            return self.last_lr

        learning_rate.get_lr = types.MethodType(get_lr, learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class CosineFixmatch(LRBase):
    """Cosine decay in FixMatch style

    Args:
        epochs (int): total epoch(s)
        step_each_epoch (int): number of iterations within an epoch
        learning_rate (float): learning rate
        num_warmup_steps (int): the number warmup steps.
        warmunum_cycles (float, optional): the factor for cosine in FixMatch learning rate. Defaults to 7 / 16.
        last_epoch (int, optional): last epoch. Defaults to -1.
        by_epoch (bool, optional): learning rate decays by epoch when by_epoch is True, else by iter. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 learning_rate,
                 num_warmup_steps,
                 num_cycles=7 / 16,
                 last_epoch=-1,
                 by_epoch=False):
        self.epochs = epochs
        self.step_each_epoch = step_each_epoch
        self.learning_rate = learning_rate
        self.num_warmup_steps = num_warmup_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch
        self.by_epoch = by_epoch

    def __call__(self):
        def _lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(
                    max(1, self.num_warmup_steps))
            no_progress = float(current_step - self.num_warmup_steps) / \
                        float(max(1, self.epochs * self.step_each_epoch - self.num_warmup_steps))
            return max(0., math.cos(math.pi * self.num_cycles * no_progress))

        learning_rate = lr.LambdaDecay(
            learning_rate=self.learning_rate,
            lr_lambda=_lr_lambda,
            last_epoch=self.last_epoch)
        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate


class OneCycleLR(LRBase):
    """OneCycleLR learning rate decay

    Args:
        epochs (int): Total epoch(s).
        step_each_epoch (int): Number of iterations within an epoch.
        max_learning_rate (float): Maximum learning rate in the cycle. It defines the cycle amplitude as above.
            Since there is some scaling operation during process of learning rate adjustment,
            max_learning_rate may not actually be reached.
        warmup_epoch (int): Number of warmup epoch(s).
        warmup_start_lr (float): Start learning rate within warmup.
        divide_factor (float, optional): Initial learning rate will be determined by initial_learning_rate = max_learning_rate / divide_factor. Default: 25.
        phase_pct (float): The percentage of total steps which used to increasing learning rate. Default: 0.3.
        end_learning_rate (float, optional): The minimum learning rate during training, it should be much less than initial learning rate.
        anneal_strategy (str, optional): Strategy of adjusting learning rate.'cos' for cosine annealing, 'linear' for linear annealing. Default: 'cos'.
        three_phase (bool, optional): Whether to use three-phase.
        last_epoch (int, optional): The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
        by_epoch (bool): Learning rate decays by epoch when by_epoch is True, else by iter.
        verbose: (bool, optional): If True, prints a message to stdout for each update. Defaults to False.
    """

    def __init__(self,
                 epochs,
                 step_each_epoch,
                 max_learning_rate,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 divide_factor=25.0,
                 end_learning_rate=1e-10,
                 phase_pct=0.3,
                 anneal_strategy='cos',
                 three_phase=False,
                 by_epoch=False,
                 last_epoch=-1,
                 verbose=False):
        super().__init__(
            epochs, step_each_epoch, max_learning_rate,
            warmup_epoch, warmup_start_lr, last_epoch, by_epoch, verbose)
        self.max_learning_rate = max_learning_rate
        self.total_steps = epochs * step_each_epoch
        self.divide_factor = divide_factor
        self.end_learning_rate = end_learning_rate
        self.phase_pct = phase_pct
        self.anneal_strategy = anneal_strategy
        self.three_phase = three_phase
        self.last_epoch = last_epoch

    def __call__(self):
        learning_rate = lr.OneCycleLR(
            max_learning_rate=self.max_learning_rate,
            total_steps=self.total_steps,
            end_learning_rate=self.end_learning_rate,
            divide_factor=self.divide_factor,
            phase_pct=self.phase_pct,
            anneal_strategy=self.anneal_strategy,
            three_phase=self.three_phase,
            last_epoch=self.last_epoch,
            verbose=self.verbose)

        if self.warmup_steps > 0:
            learning_rate = self.linear_warmup(learning_rate)

        setattr(learning_rate, "by_epoch", self.by_epoch)
        return learning_rate
