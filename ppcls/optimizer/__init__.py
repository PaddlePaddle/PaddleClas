# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import paddle

from ppcls.utils import logger

from . import optimizer

__all__ = ['build_optimizer']


def build_lr_scheduler(lr_config, epochs, step_each_epoch):
    from . import learning_rate
    lr_config.update({'epochs': epochs, 'step_each_epoch': step_each_epoch})
    if 'name' in lr_config:
        lr_name = lr_config.pop('name')
        lr = getattr(learning_rate, lr_name)(**lr_config)
        if isinstance(lr, paddle.optimizer.lr.LRScheduler):
            return lr
        else:
            return lr()
    else:
        lr = lr_config['learning_rate']
    return lr


# model_list is None in static graph
def build_optimizer(config, epochs, step_each_epoch, model_list=None):
    config = copy.deepcopy(config)
    optim = []
    lr = []

    if 'name' in config:
        config = {'model': config}  # only one config for model

    for cfg_idx, (cfg_name, cfg_content) in enumerate(config.items()):
        # step1 build lr
        lr_ = build_lr_scheduler(
            cfg_content.pop('lr'), epochs, step_each_epoch)
        logger.debug("build {} lr ({}) success..".format(cfg_name, lr_))
        # step2 build regularization
        if 'regularizer' in cfg_content and cfg_content[
                'regularizer'] is not None:
            if 'weight_decay' in cfg_content:
                logger.warning(
                    "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
                )
            reg_config = cfg_content.pop('regularizer')
            reg_name = reg_config.pop('name') + 'Decay'
            reg = getattr(paddle.regularizer, reg_name)(**reg_config)
            cfg_content["weight_decay"] = reg
            logger.debug("build {} regularizer ({}) success..".format(cfg_name,
                                                                      reg))
        # step3 build optimizer
        optim_name = cfg_content.pop('name')
        if 'clip_norm' in cfg_content:
            clip_norm = cfg_content.pop('clip_norm')
            grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        else:
            grad_clip = None
        optim_ = getattr(optimizer, optim_name)(
            learning_rate=lr_, grad_clip=grad_clip,
            **cfg_content)(model_list=model_list[cfg_idx:cfg_idx + 1])

        optim.append(optim_)
        lr.append(lr_)
        logger.debug("build {} optimizer ({}) success..".format(cfg_name,
                                                                optim))
    return optim, lr
