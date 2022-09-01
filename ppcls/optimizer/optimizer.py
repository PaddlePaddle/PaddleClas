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

import inspect

from paddle import optimizer as optim
from paddle import _C_ops
from paddle.fluid import core, framework
from paddle.fluid.framework import _in_legacy_dygraph, in_dygraph_mode
from paddle.fluid.regularizer import L2DecayRegularizer
from ppcls.utils import logger


class SGD(object):
    """
    Args:
    learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
        It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
    parameters (list|tuple, optional): List/Tuple of ``Tensor`` to update to minimize ``loss``. \
        This parameter is required in dygraph mode. \
        The default value is None in static mode, at this time all parameters will be updated.
    weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
        It canbe a float value as coeff of L2 regularization or \
        :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
        If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
        the regularization setting here in optimizer will be ignored for this parameter. \
        Otherwise, the regularization setting here in optimizer will take effect. \
        Default None, meaning there is no regularization.
    grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
        some derived class of ``GradientClipBase`` . There are three cliping strategies
        ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
        :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
    name (str, optional): The default value is None. Normally there is no need for user
            to set this property.
    """

    def __init__(self,
                 learning_rate=0.001,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False,
                 name=None):
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.multi_precision = multi_precision
        self.name = name

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        argspec = inspect.getargspec(optim.SGD.__init__).args
        if 'multi_precision' in argspec:
            opt = optim.SGD(learning_rate=self.learning_rate,
                            parameters=parameters,
                            weight_decay=self.weight_decay,
                            grad_clip=self.grad_clip,
                            multi_precision=self.multi_precision,
                            name=self.name)
        else:
            opt = optim.SGD(learning_rate=self.learning_rate,
                            parameters=parameters,
                            weight_decay=self.weight_decay,
                            grad_clip=self.grad_clip,
                            name=self.name)
        return opt


class _Momentum(optim.Momentum):
    def __init__(self, *args, apply_decay_param_fun=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._apply_decay_param_fun = apply_decay_param_fun

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)
        if isinstance(param_and_grad, dict):
            param_and_grad = self._update_param_group(param_and_grad)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])
        lr = self._create_param_lr(param_and_grad)

        # For fusion of momentum and l2decay 
        param = param_and_grad[0]
        regularization_method = self._regularization_method
        regularization_coeff = self._regularization_coeff
        if hasattr(param, 'regularizer'):
            # we skip param's l2decay before, so fuse it with momentum here.
            if isinstance(param.regularizer, L2DecayRegularizer):
                regularization_method = "l2_decay"
                regularization_coeff = param.regularizer._regularization_coeff
            # the param's regularization has been done before, we avoid do l2decay in momentum.
            elif param.regularizer is not None:
                regularization_method = ""
                regularization_coeff = 0.0

        #######################################################################

        # Whether we should do weight decay for the parameter.
        if self._apply_decay_param_fun is not None \
                and not self._apply_decay_param_fun(param_and_grad[0].name):
            regularization_method = ""
            regularization_coeff = 0.0

        #######################################################################

        find_master = self._multi_precision and param_and_grad[
            0].dtype == core.VarDesc.VarType.FP16
        master_weight = (self._master_weights[param_and_grad[0].name]
                         if find_master else None)

        if _in_legacy_dygraph():
            if isinstance(param_and_grad, dict):
                self._update_regularization(param_and_grad['weight_decay'])
            _, _, _ = _C_ops.momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, param_and_grad[0], velocity_acc, master_weight,
                'mu', self._momentum, 'use_nesterov', self._use_nesterov,
                'regularization_method', regularization_method,
                'regularization_coeff', regularization_coeff,
                'multi_precision', find_master)
            return None
        if in_dygraph_mode():
            if isinstance(param_and_grad, dict):
                self._update_regularization(param_and_grad['weight_decay'])
            return _C_ops.final_state_momentum(
                param_and_grad[0], param_and_grad[1], velocity_acc, lr,
                master_weight, self._momentum, self._use_nesterov,
                regularization_method, regularization_coeff, find_master,
                self._rescale_grad)

        attrs = {
            "mu": self._momentum,
            "use_nesterov": self._use_nesterov,
            "regularization_method": regularization_method,
            "regularization_coeff": regularization_coeff,
            "multi_precision": find_master,
            "rescale_grad": self._rescale_grad
        }

        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [velocity_acc],
            "LearningRate": [lr]
        }

        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }

        if find_master:
            inputs["MasterParam"] = master_weight
            outputs["MasterParamOut"] = master_weight

        # create the momentum optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op


class Momentum(object):
    """
    Simple Momentum optimizer with velocity state.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum,
                 use_nesterov=False,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=True,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.use_nesterov = use_nesterov
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None

        # TODO(gaotingquan): model_list is None when in static graph, "no_weight_decay" not work.
        if model_list is None:
            if self.one_dim_param_no_weight_decay or len(
                    self.no_weight_decay_name_list) != 0:
                msg = "\"Momentum\" does not support setting \"no_weight_decay\" in static graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = [
            p.name for model in model_list for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ] if model_list else []

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name
                for model in model_list for n, p in model.named_parameters()
                if len(p.shape) == 1
            ] if model_list else []

        opt = _Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            use_nesterov=self.use_nesterov,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            multi_precision=self.multi_precision,
            parameters=parameters,
            apply_decay_param_fun=self._apply_decay_param_fun)
        if hasattr(opt, '_use_multi_tensor'):
            opt = _Momentum(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                use_nesterov=self.use_nesterov,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
                multi_precision=self.multi_precision,
                parameters=parameters,
                use_multi_tensor=True,
                apply_decay_param_fun=self._apply_decay_param_fun)
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list


class Adam(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-08,
                 parameter_list=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None,
                 lazy_mode=False,
                 multi_precision=False):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.parameter_list = parameter_list
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.name = name
        self.lazy_mode = lazy_mode
        self.multi_precision = multi_precision

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        opt = optim.Adam(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            name=self.name,
            lazy_mode=self.lazy_mode,
            multi_precision=self.multi_precision,
            parameters=parameters)
        return opt


class RMSProp(object):
    """
    Root Mean Squared Propagation (RMSProp) is an unpublished, adaptive learning rate method.
    Args:
        learning_rate (float|Variable) - The learning rate used to update parameters.
            Can be a float value or a Variable with one float value as data element.
        momentum (float) - Momentum factor.
        rho (float) - rho value in equation.
        epsilon (float) - avoid division by zero, default is 1e-6.
        regularization (WeightDecayRegularizer, optional) - The strategy of regularization.
    """

    def __init__(self,
                 learning_rate,
                 momentum=0.0,
                 rho=0.95,
                 epsilon=1e-6,
                 weight_decay=None,
                 grad_clip=None,
                 multi_precision=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None
        opt = optim.RMSProp(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            rho=self.rho,
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            parameters=parameters)
        return opt


class AdamW(object):
    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=None,
                 multi_precision=False,
                 grad_clip=None,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False,
                 **args):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.multi_precision = multi_precision
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = sum([m.parameters() for m in model_list],
                         []) if model_list else None

        # TODO(gaotingquan): model_list is None when in static graph, "no_weight_decay" not work.
        if model_list is None:
            if self.one_dim_param_no_weight_decay or len(
                    self.no_weight_decay_name_list) != 0:
                msg = "\"AdamW\" does not support setting \"no_weight_decay\" in static graph. Please use dynamic graph."
                logger.error(Exception(msg))
                raise Exception(msg)

        self.no_weight_decay_param_name_list = [
            p.name for model in model_list for n, p in model.named_parameters()
            if any(nd in n for nd in self.no_weight_decay_name_list)
        ] if model_list else []

        if self.one_dim_param_no_weight_decay:
            self.no_weight_decay_param_name_list += [
                p.name
                for model in model_list for n, p in model.named_parameters()
                if len(p.shape) == 1
            ] if model_list else []

        opt = optim.AdamW(
            learning_rate=self.learning_rate,
            beta1=self.beta1,
            beta2=self.beta2,
            epsilon=self.epsilon,
            parameters=parameters,
            weight_decay=self.weight_decay,
            multi_precision=self.multi_precision,
            grad_clip=self.grad_clip,
            apply_decay_param_fun=self._apply_decay_param_fun)
        return opt

    def _apply_decay_param_fun(self, name):
        return name not in self.no_weight_decay_param_name_list
