# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

import errno
import os
from typing import Any, Dict, List

import paddle
import paddle.nn as nn
from ppcls.utils import logger
from ppcls.utils.dist_utils import main_only
from paddle.optimizer import Optimizer
from .download import get_weights_path_from_url

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path: str) -> None:
    """
    mkdir if not exists, ignore the exception when multiprocess mkdir together
    """
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(path):
                logger.warning(
                    'be happy if some process has already created {}'.format(
                        path))
            else:
                raise OSError('Failed to mkdir {}'.format(path))


def load_dygraph_pretrain(models: nn.LayerList, path: str=None) -> None:
    """load pretrained parameters file to models

    Args:
        models (nn.LayerList): models which pretrained parameters loaded to.
        path (str, optional): pretrained parameters file path. Defaults to None.

    """
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {} does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    models.set_dict(param_state_dict)
    return


def load_dygraph_pretrain_from_url(models: nn.LayerList,
                                   pretrained_url: str,
                                   use_ssld: bool=False) -> None:
    """load pretrained parameters from online url to models

    Args:
        models (nn.LayerList): models which pretrained parameters loaded to.
        pretrained_url (str): pretrained weight's url.
        use_ssld (bool, optional): if download ssld parameters. Defaults to False.
    """
    if use_ssld:
        pretrained_url = pretrained_url.replace("_pretrained",
                                                "_ssld_pretrained")
    local_weight_path = get_weights_path_from_url(pretrained_url).replace(
        ".pdparams", "")
    load_dygraph_pretrain(models, path=local_weight_path)
    return


def load_distillation_model(model, pretrained_model):
    logger.info("In distillation mode, teacher model will be "
                "loaded firstly before student model.")

    if not isinstance(pretrained_model, list):
        pretrained_model = [pretrained_model]

    teacher = model.teacher if hasattr(model,
                                       "teacher") else model._layers.teacher
    student = model.student if hasattr(model,
                                       "student") else model._layers.student
    load_dygraph_pretrain(teacher, path=pretrained_model[0])
    logger.info("Finish initing teacher model from {}".format(
        pretrained_model))
    # load student model
    if len(pretrained_model) >= 2:
        load_dygraph_pretrain(student, path=pretrained_model[1])
        logger.info("Finish initing student model from {}".format(
            pretrained_model))


def init_model(config: dict,
               models: nn.LayerList,
               optimizers: List[Optimizer]=None) -> Dict[str, Any]:
    """init model with given model and optimizer's parameters.

    Args:
        config (AttrDict): config dict.
        models (nn.LayerList): models
        optimizers (List[Optimizer], optional): optimizers. Defaults to None.

        _type_: _description_
    """
    checkpoints = config.get('checkpoints')
    if checkpoints and optimizers is not None:
        assert os.path.exists(checkpoints + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        para_dict = paddle.load(checkpoints + ".pdparams")
        opt_dict = paddle.load(checkpoints + ".pdopt")
        metric_dict = paddle.load(checkpoints + ".pdstates")
        models.set_dict(para_dict)
        if isinstance(opt_dict, list):
            for opt_ind, sub_opt_dict in enumerate(opt_dict):
                optimizers[opt_ind].set_state_dict(sub_opt_dict)
        else:
            optimizers.set_state_dict(opt_dict)
        logger.info("Finish load checkpoints from {}".format(checkpoints))
        return metric_dict

    pretrained_model = config.get('pretrained_model')
    use_distillation = config.get('use_distillation', False)
    if pretrained_model:
        if use_distillation:
            load_distillation_model(models, pretrained_model)
        else:  # common load
            load_dygraph_pretrain(models, path=pretrained_model)
            logger.info(
                logger.coloring("Finish load pretrained model from {}".format(
                    pretrained_model), "HEADER"))


@main_only
def save_model(nets,
               optimizers,
               metric_info,
               model_path,
               model_name="",
               prefix='ppcls'):
    """
    save model to the target path
    """
    model_path = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_path)
    model_path = os.path.join(model_path, prefix)
    paddle.save(nets.state_dict(), model_path + ".pdparams")
    paddle.save([opt.state_dict() for opt in optimizers],
                model_path + ".pdopt")

    paddle.save(metric_info, model_path + ".pdstates")
    logger.info("Already save model in {}".format(model_path))
