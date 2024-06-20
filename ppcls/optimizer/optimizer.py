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

import inspect
import paddle
from paddle import optimizer as optim
from ppcls.utils import logger
from functools import partial


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
        argspec = inspect.getfullargspec(optim.SGD.__init__).args
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
                 weight_decay=None,
                 grad_clip=None,
                 use_nesterov=False,
                 multi_precision=True,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.multi_precision = multi_precision
        self.use_nesterov = use_nesterov
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = None
        if model_list:
            # TODO(gaotingquan): to avoid cause issues for unset no_weight_decay models
            if len(self.no_weight_decay_name_list) > 0:
                params_with_decay = []
                params_without_decay = []
                for m in model_list:
                    for n, p in m.named_parameters():
                        if any(nd in n for nd in self.no_weight_decay_name_list) \
                            or (self.one_dim_param_no_weight_decay and len(p.shape) == 1):
                            params_without_decay.append(p)
                        else:
                            params_with_decay.append(p)
                parameters = [{
                    "params": params_with_decay,
                    "weight_decay": self.weight_decay
                }, {
                    "params": params_without_decay,
                    "weight_decay": 0.0
                }]
            else:
                parameters = sum([m.parameters() for m in model_list], [])
        opt = optim.Momentum(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
            grad_clip=self.grad_clip,
            multi_precision=self.multi_precision,
            use_nesterov=self.use_nesterov,
            parameters=parameters)
        if hasattr(opt, '_use_multi_tensor'):
            opt = optim.Momentum(
                learning_rate=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                grad_clip=self.grad_clip,
                multi_precision=self.multi_precision,
                parameters=parameters,
                use_nesterov=self.use_nesterov,
                use_multi_tensor=True)
        return opt


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
                 multi_precision=False,
                 no_weight_decay_name=None,
                 one_dim_param_no_weight_decay=False):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.no_weight_decay_name_list = no_weight_decay_name.split(
        ) if no_weight_decay_name else []
        self.one_dim_param_no_weight_decay = one_dim_param_no_weight_decay

    def __call__(self, model_list):
        # model_list is None in static graph
        parameters = None
        if model_list:
            params_with_decay = []
            params_without_decay = []
            for m in model_list:
                for n, p in m.named_parameters():
                    if any(nd in n for nd in self.no_weight_decay_name_list) \
                        or (self.one_dim_param_no_weight_decay and len(p.shape) == 1):
                        params_without_decay.append(p)
                    else:
                        params_with_decay.append(p)
            if params_without_decay:
                parameters = [{
                    "params": params_with_decay,
                    "weight_decay": self.weight_decay
                }, {
                    "params": params_without_decay,
                    "weight_decay": 0.0
                }]
            else:
                parameters = params_with_decay
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


class AdamWDL(object):
    """
    The AdamWDL optimizer is implemented based on the AdamW Optimization with dynamic lr setting.
    Generally it's used for transformer model.
    """

    def __init__(self,
                 learning_rate=0.001,
                 beta1=0.9,
                 beta2=0.999,
                 epsilon=1e-8,
                 weight_decay=None,
                 multi_precision=False,
                 grad_clip=None,
                 layerwise_decay=None,
                 filter_bias_and_bn=True,
                 **args):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.grad_clip = grad_clip
        self.weight_decay = weight_decay
        self.multi_precision = multi_precision
        self.layerwise_decay = layerwise_decay
        self.filter_bias_and_bn = filter_bias_and_bn

    class AdamWDLImpl(optim.AdamW):
        def __init__(self,
                     learning_rate=0.001,
                     beta1=0.9,
                     beta2=0.999,
                     epsilon=1e-8,
                     parameters=None,
                     weight_decay=0.01,
                     apply_decay_param_fun=None,
                     grad_clip=None,
                     lazy_mode=False,
                     multi_precision=False,
                     layerwise_decay=1.0,
                     n_layers=12,
                     name_dict=None,
                     name=None):
            if not isinstance(layerwise_decay, float) and \
                    not isinstance(layerwise_decay, paddle.static.Variable):
                raise TypeError("coeff should be float or Tensor.")
            self.layerwise_decay = layerwise_decay
            self.name_dict = name_dict
            self.n_layers = n_layers
            self._coeff = weight_decay
            self._lr_to_coeff = dict()
            self.set_param_lr_func = partial(
                self._layerwise_lr_decay, layerwise_decay, name_dict, n_layers)
            super().__init__(
                learning_rate=learning_rate,
                parameters=parameters,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                grad_clip=grad_clip,
                name=name,
                apply_decay_param_fun=apply_decay_param_fun,
                weight_decay=weight_decay,
                lazy_mode=lazy_mode,
                multi_precision=multi_precision,)

        # Layerwise decay
        def _layerwise_lr_decay(self, decay_rate, name_dict, n_layers, param):
            """
            Args:
                decay_rate (float):
                    The layer-wise decay ratio.
                name_dict (dict):
                    The keys of name_dict is dynamic name of model while the value
                    of name_dict is static name.
                    Use model.named_parameters() to get name_dict.
                n_layers (int):
                    Total number of layers in the transformer encoder.
            """
            ratio = 1.0
            static_name = name_dict[param.name]
            if "blocks" in static_name:
                idx = static_name.find("blocks.")
                layer = int(static_name[idx:].split(".")[1])
                ratio = decay_rate**(n_layers - layer)
            elif any([
                    key in static_name
                    for key in ["embed", "token", "conv1", "ln_pre"]
            ]):
                ratio = decay_rate**(n_layers + 1)
            # param.optimize_attr["learning_rate"] *= ratio
            return ratio
        def _append_decoupled_weight_decay(self, block, param_and_grad):
            """
            Add decoupled weight decay op.
                parameter = parameter - parameter * coeff * lr
            Args:
                block: block in which variable is to be created
                param_and_grad: (parameters, gradients) pairs,
                    the parameters need to decay.
            Raises:
                Exception: The type of coeff and parameter is not consistent.
            """
            if isinstance(param_and_grad, dict):
                param_and_grad = self._update_param_group(param_and_grad)
            param, grad = param_and_grad

            if self._apply_decay_param_fun is not None and not self._apply_decay_param_fun(param.name):
                return

            if isinstance(self._learning_rate, float):
                learning_rate = self._learning_rate
            else:
                # NOTE. We add this function to the _append_optimize_op(),
                # for we must make sure _create_param_lr() be called after
                # optimizer._create_global_learning_rate().
                learning_rate = self._create_param_lr(param_and_grad)

            with block.program._optimized_guard([param, grad]), paddle.static.name_scope("weight decay"):
                self._params_name.add(param.name)

                # If it has been calculated, the result will be reused.
                # NOTE(wangxi): In dygraph mode, apply_gradient will be executed
                # every step, so need clear _lr_to_coeff every step,
                # we do this in _create_optimization_pass
                decay_coeff = self._lr_to_coeff.get(learning_rate, None)
                if decay_coeff is None:
                    # NOTE(wangxi): for pipeline to set device:all
                    with paddle.static.device_guard(None):
                        decay_coeff = 1.0 - learning_rate * self._coeff
                    self._lr_to_coeff[learning_rate] = decay_coeff

                find_master = self._multi_precision and param.dtype == paddle.float16
                if find_master:
                    master_weight = self._master_weights[param.name]
                    scaled_param = master_weight * decay_coeff
                    paddle.assign(scaled_param, output=master_weight)
                else:
                    scaled_param = param * decay_coeff
                    paddle.assign(scaled_param, output=param)

        def _append_optimize_op(self, block, param_and_grad):
            if self.set_param_lr_func is None:
                return super()._append_optimize_op(block, param_and_grad)

            self._append_decoupled_weight_decay(block, param_and_grad)
            prev_lr = param_and_grad[0].optimize_attr["learning_rate"]
            ratio = self.set_param_lr_func(param_and_grad[0])
            param_and_grad[0].optimize_attr["learning_rate"] *= ratio

            # excute Adam op
            res = super()._append_optimize_op(block, param_and_grad)
            param_and_grad[0].optimize_attr["learning_rate"] = prev_lr
            return res

    def __call__(self, model_list):
        model = model_list[0]
        if self.weight_decay and self.filter_bias_and_bn:
            skip = {}
            if hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            decay_dict = {
                param.name: not (len(param.shape) == 1 or
                                 name.endswith(".bias") or name in skip)
                for name, param in model.named_parameters()
                if not 'teacher' in name
            }
            parameters = [
                param for param in model.parameters()
                if 'teacher' not in param.name
            ]
            weight_decay = 0.
        else:
            parameters = model.parameters()
        opt_args = dict(
            learning_rate=self.learning_rate, weight_decay=self.weight_decay)
        opt_args['parameters'] = parameters
        if decay_dict is not None:
            opt_args['apply_decay_param_fun'] = lambda n: decay_dict[n]
        opt_args['epsilon'] = self.epsilon
        opt_args['beta1'] = self.beta1
        opt_args['beta2'] = self.beta2
        if self.layerwise_decay and self.layerwise_decay < 1.0:
            opt_args['layerwise_decay'] = self.layerwise_decay
            name_dict = dict()
            for n, p in model.named_parameters():
                name_dict[p.name] = n
            opt_args['name_dict'] = name_dict
            opt_args['n_layers'] = model.get_num_layers()
        optimizer = self.AdamWDLImpl(**opt_args)

        return optimizer