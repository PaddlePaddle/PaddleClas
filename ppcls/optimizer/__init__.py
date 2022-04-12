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
    if 'name' in config:
        # NOTE: build optimizer and lr for model only.
        # step1 build lr
        lr = build_lr_scheduler(config.pop('lr'), epochs, step_each_epoch)
        logger.debug("build model's lr ({}) success..".format(lr))
        # step2 build regularization
        if 'regularizer' in config and config['regularizer'] is not None:
            if 'weight_decay' in config:
                logger.warning(
                    "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
                )
            reg_config = config.pop('regularizer')
            reg_name = reg_config.pop('name') + 'Decay'
            reg = getattr(paddle.regularizer, reg_name)(**reg_config)
            config["weight_decay"] = reg
            logger.debug("build model's regularizer ({}) success..".format(
                reg))
        # step3 build optimizer
        optim_name = config.pop('name')
        if 'clip_norm' in config:
            clip_norm = config.pop('clip_norm')
            grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        else:
            grad_clip = None
        optim = getattr(optimizer, optim_name)(
            learning_rate=lr, grad_clip=grad_clip,
            **config)(model_list=model_list[0:1])
        optim = [optim, ]
        lr = [lr, ]
        logger.debug("build model's optimizer ({}) success..".format(optim))
    else:
        # NOTE: build optimizer and lr for model and loss.
        config_model = config['model']
        config_loss = config['loss']
        # step1 build lr
        lr_model = build_lr_scheduler(
            config_model.pop('lr'), epochs, step_each_epoch)
        logger.debug("build model's lr ({}) success..".format(lr_model))
        # step2 build regularization
        if 'regularizer' in config_model and config_model[
                'regularizer'] is not None:
            if 'weight_decay' in config_model:
                logger.warning(
                    "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
                )
            reg_config = config_model.pop('regularizer')
            reg_name = reg_config.pop('name') + 'Decay'
            reg_model = getattr(paddle.regularizer, reg_name)(**reg_config)
            config_model["weight_decay"] = reg_model
            logger.debug("build model's regularizer ({}) success..".format(
                reg_model))
        # step3 build optimizer
        optim_name = config_model.pop('name')
        if 'clip_norm' in config_model:
            clip_norm = config_model.pop('clip_norm')
            grad_clip_model = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        else:
            grad_clip_model = None
        optim_model = getattr(optimizer, optim_name)(
            learning_rate=lr_model, grad_clip=grad_clip_model,
            **config_model)(model_list=model_list[0:1])

        # step4 build lr for loss
        lr_loss = build_lr_scheduler(
            config_loss.pop('lr'), epochs, step_each_epoch)
        logger.debug("build loss's lr ({}) success..".format(lr_loss))
        # step5 build regularization for loss
        if 'regularizer' in config_loss and config_loss[
                'regularizer'] is not None:
            if 'weight_decay' in config_loss:
                logger.warning(
                    "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
                )
            reg_config = config_loss.pop('regularizer')
            reg_name = reg_config.pop('name') + 'Decay'
            reg_loss = getattr(paddle.regularizer, reg_name)(**reg_config)
            config_loss["weight_decay"] = reg_loss
            logger.debug("build loss's regularizer ({}) success..".format(
                reg_loss))
        # step6 build optimizer for loss
        optim_name = config_loss.pop('name')
        if 'clip_norm' in config_loss:
            clip_norm = config_loss.pop('clip_norm')
            grad_clip_loss = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        else:
            grad_clip_loss = None
        optim_loss = getattr(optimizer, optim_name)(
            learning_rate=lr_loss, grad_clip=grad_clip_loss,
            **config_loss)(model_list=model_list[1:2])

        optim = [optim_model, optim_loss]
        lr = [lr_model, lr_loss]
        logger.debug("build loss's optimizer ({}) success..".format(optim))
    return optim, lr
