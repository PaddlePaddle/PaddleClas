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

import datetime
from ppcls.utils import logger
from ppcls.utils.misc import AverageMeter
import numpy as np


def update_metric(trainer, out, batch, batch_size):
    # calc metric
    if trainer.train_metric_func is not None:
        metric_dict = trainer.train_metric_func(out, batch[-1])
        for key in metric_dict:
            if key not in trainer.output_info:
                trainer.output_info[key] = AverageMeter(key, '7.5f')
            trainer.output_info[key].update(
                float(metric_dict[key]), batch_size)


def update_loss(trainer, loss_dict, batch_size):
    # update_output_info
    for key in loss_dict:
        if key not in trainer.output_info:
            trainer.output_info[key] = AverageMeter(key, '7.5f')
        trainer.output_info[key].update(float(loss_dict[key]), batch_size)


def log_info(trainer, batch_size, epoch_id, iter_id):
    lr_msg = ", ".join([
        "lr({}): {:.8f}".format(type_name(lr), lr.get_lr())
        for i, lr in enumerate(trainer.lr_sch)
    ])
    metric_msg = ", ".join([
        "{}: {:.5f}".format(key, trainer.output_info[key].avg)
        for key in trainer.output_info
    ])
    time_msg = "s, ".join([
        "{}: {:.5f}".format(key, trainer.time_info[key].avg)
        for key in trainer.time_info
    ])

    ips_msg = "ips: {:.5f} samples/s".format(
        batch_size / trainer.time_info["batch_cost"].avg)

    eta_sec = (
        (trainer.config["Global"]["epochs"] - epoch_id + 1) *
        trainer.iter_per_epoch - iter_id) * trainer.time_info["batch_cost"].avg
    eta_msg = "eta: {:s}".format(str(datetime.timedelta(seconds=int(eta_sec))))
    logger.info("[Train][Epoch {}/{}][Iter: {}/{}]{}, {}, {}, {}, {}".format(
        epoch_id, trainer.config["Global"]["epochs"], iter_id, trainer.
        iter_per_epoch, lr_msg, metric_msg, time_msg, ips_msg, eta_msg))

    for i, lr in enumerate(trainer.lr_sch):
        logger.scaler(
            name="lr({})".format(type_name(lr)),
            value=lr.get_lr(),
            step=trainer.global_step,
            writer=trainer.vdl_writer)
    for key in trainer.output_info:
        logger.scaler(
            name="train_{}".format(key),
            value=trainer.output_info[key].avg,
            step=trainer.global_step,
            writer=trainer.vdl_writer)


def type_name(object: object) -> str:
    """get class name of an object"""
    return object.__class__.__name__


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def cancel_gradients_last_layer(epoch, model, freeze_last_layer):
    if epoch >= freeze_last_layer:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            # can not use `stop_gradient`
            p.clear_grad()
