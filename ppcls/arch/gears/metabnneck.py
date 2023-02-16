# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

from collections import defaultdict
import copy
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..utils import get_param_attr_dict


class MetaBN1D(nn.BatchNorm1D):
    def forward(self, inputs, opt={}):
        mode = opt.get("bn_mode", "general") if self.training else "eval"
        if mode == "general":  # update, but not apply running_mean/var
            result = F.batch_norm(inputs, self._mean, self._variance,
                                  self.weight, self.bias, self.training,
                                  self._momentum, self._epsilon)
        elif mode == "hold":  # not update, not apply running_mean/var
            result = F.batch_norm(
                inputs,
                paddle.mean(
                    inputs, axis=0),
                paddle.var(inputs, axis=0),
                self.weight,
                self.bias,
                self.training,
                self._momentum,
                self._epsilon)
        elif mode == "eval":  # fix and apply running_mean/var,
            if self._mean is None:
                result = F.batch_norm(
                    inputs,
                    paddle.mean(
                        inputs, axis=0),
                    paddle.var(inputs, axis=0),
                    self.weight,
                    self.bias,
                    True,
                    self._momentum,
                    self._epsilon)
            else:
                result = F.batch_norm(inputs, self._mean, self._variance,
                                      self.weight, self.bias, False,
                                      self._momentum, self._epsilon)
        return result


class MetaBNNeck(nn.Layer):
    def __init__(self, num_features, **kwargs):
        super(MetaBNNeck, self).__init__()
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.Constant(value=0.0),
            trainable=False)

        if 'weight_attr' in kwargs:
            weight_attr = get_param_attr_dict(kwargs['weight_attr'])

        bias_attr = None
        if 'bias_attr' in kwargs:
            bias_attr = get_param_attr_dict(kwargs['bias_attr'])

        use_global_stats = None
        if 'use_global_stats' in kwargs:
            use_global_stats = get_param_attr_dict(kwargs['use_global_stats'])

        self.feat_bn = MetaBN1D(
            num_features,
            momentum=0.9,
            epsilon=1e-05,
            weight_attr=weight_attr,
            bias_attr=bias_attr,
            use_global_stats=use_global_stats)
        self.flatten = nn.Flatten()
        self.opt = {}

    def forward(self, x):
        x = self.flatten(x)
        x = self.feat_bn(x, self.opt)
        return x

    def reset_opt(self):
        self.opt = defaultdict()

    def setup_opt(self, opt):
        """
        Arg:
            opt (dict): Optional setting to change the behavior of MetaBIN during training. 
                It includes three settings which are `enable_inside_update`, `lr_gate` and `bn_mode`.
        """
        self.check_opt(opt)
        self.opt = copy.deepcopy(opt)

    @classmethod
    def check_opt(cls, opt):
        assert isinstance(opt, dict), \
            TypeError('Got the wrong type of `opt`. Please use `dict` type.')

        if opt.get('enable_inside_update', False) and 'lr_gate' not in opt:
            raise RuntimeError('Missing `lr_gate` in opt.')

        assert isinstance(opt.get('lr_gate', 1.0), float), \
            TypeError('Got the wrong type of `lr_gate`. Please use `float` type.')
        assert isinstance(opt.get('enable_inside_update', True), bool), \
            TypeError('Got the wrong type of `enable_inside_update`. Please use `bool` type.')
        assert opt.get('bn_mode', "general") in ["general", "hold", "eval"], \
            TypeError('Got the wrong value of `bn_mode`.')
