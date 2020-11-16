#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.optimizer as pfopt
import paddle.fluid.regularizer as pfreg

__all__ = ['OptimizerBuilder']


class OptimizerBuilder(object):
    """
    Build optimizer with fluid api in fluid.layers.optimizer,
    such as fluid.layers.optimizer.Momentum()
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/optimizer_cn.html
    https://www.paddlepaddle.org.cn/documentation/docs/zh/api_cn/regularizer_cn.html

    Args:
        function(str): optimizer name of learning rate
        params(dict): parameters used for init the class
        regularizer (dict): parameters used for create regularization
    """

    def __init__(self,
                 function='Momentum',
                 params={'momentum': 0.9},
                 regularizer=None):
        self.function = function
        self.params = params
        # create regularizer
        if regularizer is not None:
            reg_func = regularizer['function'] + 'Decay'
            reg_factor = regularizer['factor']
            reg = getattr(pfreg, reg_func)(reg_factor)
            self.params['regularization'] = reg

    def __call__(self, learning_rate):
        opt = getattr(pfopt, self.function)
        return opt(learning_rate=learning_rate, **self.params)
