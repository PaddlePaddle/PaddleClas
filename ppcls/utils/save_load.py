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

import paddle
from ppcls.utils import logger
from .download import get_weights_path_from_url

__all__ = ['init_model', 'save_model', 'load_dygraph_pretrain']


def _mkdir_if_not_exist(path):
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


def _extract_student_weights(all_params, student_prefix="Student."):
    s_params = {
        key[len(student_prefix):]: all_params[key]
        for key in all_params if student_prefix in key
    }
    return s_params


def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {}.pdparams does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    if isinstance(model, list):
        for m in model:
            if hasattr(m, 'set_dict'):
                m.set_dict(param_state_dict)
    else:
        model.set_dict(param_state_dict)
    return


def load_dygraph_pretrain_from_url(model, pretrained_url, use_ssld=False):
    if use_ssld:
        pretrained_url = pretrained_url.replace("_pretrained",
                                                "_ssld_pretrained")
    local_weight_path = get_weights_path_from_url(pretrained_url).replace(
        ".pdparams", "")
    load_dygraph_pretrain(model, path=local_weight_path)
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


def init_model(config, net, optimizer=None, loss: paddle.nn.Layer=None):
    """
    load model from checkpoint or pretrained_model
    """
    checkpoints = config.get('checkpoints')
    if checkpoints and optimizer is not None:
        assert os.path.exists(checkpoints + ".pdparams"), \
            "Given dir {}.pdparams not exist.".format(checkpoints)
        assert os.path.exists(checkpoints + ".pdopt"), \
            "Given dir {}.pdopt not exist.".format(checkpoints)
        # load state dict
        opti_dict = paddle.load(checkpoints + ".pdopt")
        para_dict = paddle.load(checkpoints + ".pdparams")
        metric_dict = paddle.load(checkpoints + ".pdstates")
        # set state dict
        net.set_state_dict(para_dict)
        loss.set_state_dict(para_dict)
        for i in range(len(optimizer)):
            optimizer[i].set_state_dict(opti_dict[i] if isinstance(
                opti_dict, list) else opti_dict)
        logger.info("Finish load checkpoints from {}".format(checkpoints))
        return metric_dict

    pretrained_model = config.get('pretrained_model')
    use_distillation = config.get('use_distillation', False)
    if pretrained_model:
        if use_distillation:
            load_distillation_model(net, pretrained_model)
        else:  # common load
            load_dygraph_pretrain(net, path=pretrained_model)
            logger.info("Finish load pretrained model from {}".format(
                pretrained_model))


def save_model(net,
               optimizer,
               metric_info,
               model_path,
               model_name="",
               prefix='ppcls',
               loss: paddle.nn.Layer=None,
               save_student_model=False):
    """
    save model to the target path
    """
    if paddle.distributed.get_rank() != 0:
        return
    model_path = os.path.join(model_path, model_name)
    _mkdir_if_not_exist(model_path)
    model_path = os.path.join(model_path, prefix)

    params_state_dict = net.state_dict()
    if loss is not None:
        loss_state_dict = loss.state_dict()
        keys_inter = set(params_state_dict.keys()) & set(loss_state_dict.keys(
        ))
        assert len(keys_inter) == 0, \
            f"keys in model and loss state_dict must be unique, but got intersection {keys_inter}"
        params_state_dict.update(loss_state_dict)

    if save_student_model:
        s_params = _extract_student_weights(params_state_dict)
        if len(s_params) > 0:
            paddle.save(s_params, model_path + "_student.pdparams")

    paddle.save(params_state_dict, model_path + ".pdparams")
    paddle.save([opt.state_dict() for opt in optimizer], model_path + ".pdopt")
    paddle.save(metric_info, model_path + ".pdstates")
    logger.info("Already save model in {}".format(model_path))
