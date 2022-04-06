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

import os
import os.path as osp
import random

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.nn as nn
from ppcls.arch import build_model
from ppcls.data import build_dataloader
from ppcls.loss import build_loss
from ppcls.metric import build_metrics
from ppcls.optimizer.__init__ import build_optimizer
from ppcls.utils import logger
from ppcls.utils.save_load import (load_dygraph_pretrain,
                                   load_dygraph_pretrain_from_url)
from visualdl import LogWriter

from .config import print_config
from .logger import init_logger

__all__ = [
    'set_seed', 'set_logger', 'set_visualDL', 'set_device', 'set_dataloaders',
    'load_pretrain', 'set_amp', 'set_losses', 'set_optimizers', 'set_metrics',
    'set_distributed'
]


def set_seed(seed: int=None) -> None:
    """seed (int): random seed.

    Args:
        seed (int, optional): random seed. Defaults to None.
    """
    if seed is not None:
        assert isinstance(seed, int), "The 'seed' must be a integer!"
        paddle.seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def set_logger(engine: object, print_cfg: bool=True) -> None:
    """Set the save path for models and records.

    Args:
        engine (object): Engine object.
        print_cfg (bool, optional): Whether to print yaml configuration. Defaults to True.
    """
    engine.output_dir = engine.config.Global.output_dir
    log_file = osp.join(engine.output_dir, engine.config.Arch.name,
                        f"{engine.mode}.log")
    init_logger(log_file=log_file)
    if print_cfg:
        print_config(engine.config)


def set_visualDL(engine: object) -> None:
    """Set up the visual DL component of the paddle.

    Args:
        engine (object): Engine object.
    """
    engine.vdl_writer = None
    if dist.get_rank() == 0:
        if engine.config.Global.get('use_visualdl',
                                    False) and engine.mode == "train":
            vdl_writer_path = os.path.join(engine.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            engine.vdl_writer = LogWriter(logdir=vdl_writer_path)


def set_device(engine: object) -> None:
    """Set up the runtime device.

    Args:
        engine (object): Engine object.
    """
    engine.device = paddle.set_device(engine.config.Global.device)
    logger.info('train with paddle {} and device {}'.format(paddle.__version__,
                                                            engine.device))


def set_dataloaders(engine: object) -> None:
    """Set up the dataloader(s)
    Args:
        engine (object): Engine object.
    """
    # build dataloader
    # TODO(gaotingquan): support rec
    class_num = engine.config.Arch.get("class_num", None)
    engine.config.DataLoader.update({"class_num": class_num})
    engine.use_dali = engine.config.Global.get("use_dali", False)
    if engine.mode == 'train':
        engine.train_dataloader = build_dataloader(
            engine.config.DataLoader, "Train", engine.device, engine.use_dali)
    if engine.mode == "eval" or engine.config.Global.eval_during_train:
        if engine.eval_mode == "classification":
            engine.eval_dataloader = build_dataloader(engine.config.DataLoader,
                                                      "Eval", engine.device,
                                                      engine.use_dali)
        elif engine.eval_mode == "retrieval":
            engine.gallery_query_dataloader = None
            if len(engine.config.DataLoader.Eval.keys()) == 1:
                key = list(engine.config.DataLoader.Eval.keys())[0]
                engine.gallery_query_dataloader = build_dataloader(
                    engine.config.DataLoader.Eval, key, engine.device,
                    engine.use_dali)
            else:
                engine.gallery_dataloader = build_dataloader(
                    engine.config.DataLoader.Eval, "Gallery", engine.device,
                    engine.use_dali)
                engine.query_dataloader = build_dataloader(
                    engine.config.DataLoader.Eval, "Query", engine.device,
                    engine.use_dali)


def set_losses(engine: object) -> None:
    """Set the losses used at runtime.
    """
    if engine.mode == "train":
        loss_info = engine.config.Loss.Train
        engine.train_loss_func = build_loss(loss_info)
    if engine.mode == "eval" or (engine.mode == "train" and
                                 engine.config.Global.eval_during_train):
        loss_config = engine.config.get("Loss", None)
        if loss_config is not None:
            loss_config = loss_config.get("Eval")
            if loss_config is not None:
                engine.eval_loss_func = build_loss(loss_config)
            else:
                engine.eval_loss_func = None
        else:
            engine.eval_loss_func = None


def set_models(engine: object) -> None:
    """Set up the model.

    Args:
        engine (object): Engine object.
    """
    engine.models = nn.LayerList()
    engine.models.append(build_model(engine.config))
    if engine.mode == "train" and hasattr(engine, 'train_loss_func'):
        for loss_func in engine.train_loss_func.loss_func:
            # for components with independent parameters
            if len(loss_func.parameters()) > 0:
                engine.models.append(loss_func)
    elif engine.mode == "eval" and hasattr(engine, 'eval_loss_func'):
        for loss_func in engine.eval_loss_func.loss_func:
            # for components with independent parameters
            if len(loss_func.parameters()) > 0:
                engine.models.append(loss_func)


def load_pretrain(engine: object) -> None:
    """load pretrained model parameters.

    Args:
        engine (object): Engine object.
    """
    if engine.config.Global.pretrained_model.startswith("http"):
        load_dygraph_pretrain_from_url(engine.models,
                                       engine.config.Global.pretrained_model)
    else:
        load_dygraph_pretrain(engine.models,
                              engine.config.Global.pretrained_model)


def set_amp(engine: object) -> None:
    """set amp configurations

    Args:
        engine (object): Engine object.
    """
    engine.use_amp = False
    if "AMP" in engine.config and engine.mode == "train":
        engine.use_amp = True
        engine.scale_loss = engine.config.AMP.get("scale_loss", 1.0)
        engine.use_dynamic_loss_scaling = engine.config.AMP.get(
            "use_dynamic_loss_scaling", False)

        AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
        if paddle.is_compiled_with_cuda():
            AMP_RELATED_FLAGS_SETTING.update({
                'FLAGS_cudnn_batchnorm_spatial_persistent': 1
            })
        paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)

        engine.scaler = paddle.amp.GradScaler(
            init_loss_scaling=engine.scale_loss,
            use_dynamic_loss_scaling=engine.use_dynamic_loss_scaling)
        amp_level = engine.config.AMP.get("level", "O1")
        if amp_level not in ["O1", "O2"]:
            msg = "[Parameter Error]: The optimize level of AMP only support 'O1' and 'O2'. The level has been set to 'O1'."
            logger.warning(msg)
            engine.config.AMP.level = "O1"
            amp_level = "O1"
        engine.models, engine.optimizers = paddle.amp.decorate(
            models=engine.models,
            optimizers=engine.optimizers,
            level=amp_level,
            save_dtype='float32')


def set_optimizers(engine: object) -> None:
    """set optimizer(s)
    """
    engine.optimizers = []
    engine.lr_schs = []
    if engine.mode == 'train':
        if isinstance(engine.config.Optimizer, dict):
            # build single optimizer
            optimizer, lr_sch = build_optimizer(
                engine.config.Optimizer, engine.config.Global.epochs,
                len(engine.train_dataloader), [
                    engine.models._layers[0]
                    if isinstance(engine.models, paddle.DataParallel) else
                    engine.models[0]
                ])
            engine.optimizers.append(optimizer)
            engine.lr_schs.append(lr_sch)
        elif isinstance(engine.config.Optimizer, list):
            # build multiple optimizers
            for opt_ind, opt_cfg in enumerate(engine.config.Optimizer):
                assert len(opt_cfg.keys()) == 1, \
                    f"opt_cfg can only has one scope, but got ({opt_cfg.keys()})"
                opt_scope = list(opt_cfg.keys())[0]
                for model_ind, model in enumerate(engine.models):
                    model_name = type(model).__name__
                    if model_name == opt_scope:
                        optimizer, lr_sch = build_optimizer(
                            opt_cfg.get(opt_scope),
                            engine.config.Global.epochs,
                            len(engine.train_dataloader), [model])
                        engine.optimizers.append(optimizer)
                        engine.lr_schs.append(lr_sch)
        else:
            raise NotImplementedError(
                f"Optimizer config must be a single dict or list of dict, but got {type(engine.config.Optimizer)}"
            )


def set_metrics(engine: object) -> None:
    """Set metric(s) for model's train and eval process.
    """
    if engine.mode == 'train':
        metric_config = engine.config.get("Metric", None)
        if metric_config is not None:
            metric_config = metric_config.get("Train", None)
            if metric_config is not None:
                if getattr(engine.train_dataloader, 'collate_fn',
                           None) is not None:
                    for m_idx, m in enumerate(metric_config):
                        if "TopkAcc" in m:
                            msg = f"'TopkAcc' metric can not be used when setting 'batch_transform_ops' in config. The 'TopkAcc' metric has been removed."
                            logger.warning(msg)
                            break
                    metric_config.pop(m_idx)
                engine.train_metric_func = build_metrics(metric_config)
            else:
                engine.train_metric_func = None
    else:
        engine.train_metric_func = None

    if engine.mode == "eval" or (engine.mode == "train" and
                                 engine.config.Global.eval_during_train):
        metric_config = engine.config.get("Metric")
        if engine.eval_mode == "classification":
            if metric_config is not None:
                metric_config = metric_config.get("Eval")
                if metric_config is not None:
                    engine.eval_metric_func = build_metrics(metric_config)
        elif engine.eval_mode == "retrieval":
            if metric_config is None:
                metric_config = [{"name": "Recallk", "topk": (1, 5)}]
            else:
                metric_config = metric_config.Eval
            engine.eval_metric_func = build_metrics(metric_config)
    else:
        engine.eval_metric_func = None


def set_distributed(engine: object):
    """set distributed environment.

    Args:
        engine (object): Engine object.
    """
    world_size = dist.get_world_size()
    engine.config.Global.distributed = world_size != 1
    if engine.config.Global.distributed:
        dist.init_parallel_env()
        for i in range(len(engine.models)):
            engine.models[i] = paddle.DataParallel(engine.models[i])

    if world_size != 4 and engine.mode == "train":
        msg = f"The training strategy in config files provided by PaddleClas is based on 4 gpus. But the number of gpus is {world_size} in current training. Please modify the stategy (learning rate, batch size and so on) if use config files in PaddleClas to train."
        logger.warning(msg)
