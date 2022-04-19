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
from typing import Dict, List

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
    optim_config = config["Optimizer"]
    if isinstance(optim_config, dict):
        # convert {'name': xxx, **optim_cfg} to [{name: {scope: xxx, **optim_cfg}}]
        optim_name = optim_config.pop("name")
        optim_config: List[Dict[str, Dict]] = [{
            optim_name: {
                'scope': config["Arch"].get("name"),
                **
                optim_config
            }
        }]
    optim_list = []
    lr_list = []
    for optim_item in optim_config:
        # optim_cfg = {optim_name1: {scope: xxx, **optim_cfg}}
        # step1 build lr
        optim_name = optim_item.keys()[0]  # get optim_name1
        optim_scope = optim_item[optim_name].pop('scope')  # get scope
        optim_cfg = optim_item[optim_name]  # get optim_cfg

        lr = build_lr_scheduler(optim_cfg.pop('lr'), epochs, step_each_epoch)
        logger.debug("build lr ({}) for scope ({}) success..".format(
            lr, optim_scope))
        # step2 build regularization
        if 'regularizer' in optim_cfg and optim_cfg['regularizer'] is not None:
            if 'weight_decay' in optim_cfg:
                logger.warning(
                    "ConfigError: Only one of regularizer and weight_decay can be set in Optimizer Config. \"weight_decay\" has been ignored."
                )
            reg_config = optim_cfg.pop('regularizer')
            reg_name = reg_config.pop('name') + 'Decay'
            reg = getattr(paddle.regularizer, reg_name)(**reg_config)
            optim_cfg["weight_decay"] = reg
            logger.debug("build regularizer ({}) success..".format(reg))
        # step3 build optimizer
        if 'clip_norm' in optim_cfg:
            clip_norm = optim_cfg.pop('clip_norm')
            grad_clip = paddle.nn.ClipGradByNorm(clip_norm=clip_norm)
        else:
            grad_clip = None
        optim_model = []
        for i in range(len(model_list)):
            class_name = model_list[i].__class__.__name__
            if class_name == optim_scope:
                optim_model.append(model_list[i])
        assert len(optim_model) == 1 and len(optim_model[0].parameters()) > 0, \
            f"Invalid optim model for optim scope({optim_scope}), number of optim_model={len(optim_model)}, and number of optim_model's params={len(optim_model[0].parameters())}"
        optim = getattr(optimizer, optim_name)(
            learning_rate=lr, grad_clip=grad_clip,
            **optim_cfg)(model_list=optim_model)
        logger.debug("build optimizer ({}) for scope ({}) success..".format(
            optim, optim_scope))
        optim_list.append(optim)
        lr_list.append(lr)
    return optim_list, lr_list
