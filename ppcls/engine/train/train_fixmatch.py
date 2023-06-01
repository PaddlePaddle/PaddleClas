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

import time
import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from ppcls.utils import profiler
from paddle.nn import functional as F
import numpy as np


def train_epoch_fixmatch(engine, epoch_id, print_batch_step):
    tic = time.time()
    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.unlabel_train_dataloader_iter = iter(
            engine.unlabel_train_dataloader)
    temperture = engine.config["SSL"].get("temperture", 1)
    threshold = engine.config["SSL"].get("threshold", 0.95)
    assert engine.iter_per_epoch is not None, "Global.iter_per_epoch need to be set."
    threshold = paddle.to_tensor(threshold)
    for iter_id in range(engine.iter_per_epoch):
        if iter_id >= engine.iter_per_epoch:
            break
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        try:
            label_data_batch = engine.train_dataloader_iter.next()
        except Exception:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            label_data_batch = engine.train_dataloader_iter.next()
        try:
            unlabel_data_batch = engine.unlabel_train_dataloader_iter.next()
        except Exception:
            engine.unlabel_train_dataloader_iter = iter(
                engine.unlabel_train_dataloader)
            unlabel_data_batch = engine.unlabel_train_dataloader_iter.next()
        assert len(unlabel_data_batch) == 3
        assert unlabel_data_batch[0].shape == unlabel_data_batch[1].shape
        engine.time_info["reader_cost"].update(time.time() - tic)
        batch_size = label_data_batch[0].shape[0] + unlabel_data_batch[0].shape[0] \
            + unlabel_data_batch[1].shape[0]
        engine.global_step += 1

        # make inputs
        inputs_x, targets_x = label_data_batch
        inputs_u_w, inputs_u_s, targets_u = unlabel_data_batch
        batch_size_label = inputs_x.shape[0]
        inputs = paddle.concat([inputs_x, inputs_u_w, inputs_u_s], axis=0)

        # image input
        with engine.auto_cast(is_eval=False):
            loss_dict, logits_label = get_loss(engine, inputs,
                                               batch_size_label, temperture,
                                               threshold, targets_x)

        # loss
        loss = loss_dict["loss"]

        # backward & step opt
        scaled = engine.scaler.scale(loss)
        scaled.backward()

        for i in range(len(engine.optimizer)):
            # optimizer.step() with auto amp
            engine.scaler.step(engine.optimizer[i])
            engine.scaler.update()

        # step lr(by step)
        for i in range(len(engine.lr_sch)):
            if not getattr(engine.lr_sch[i], "by_epoch", False):
                engine.lr_sch[i].step()
        # clear grad
        for i in range(len(engine.optimizer)):
            engine.optimizer[i].clear_grad()

        # update ema
        if engine.ema:
            engine.model_ema.update(engine.model)

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, logits_label, label_data_batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()

    # step lr(by epoch)
    for i in range(len(engine.lr_sch)):
        if getattr(engine.lr_sch[i], "by_epoch", False):
            engine.lr_sch[i].step()


def get_loss(engine, inputs, batch_size_label, temperture, threshold,
             targets_x):
    # For pytroch version, inputs need to use interleave and de_interleave
    # to reshape and transpose inputs and logits, but it dosen't affect the
    # result. So this paddle version dose not use the two transpose func.
    # inputs = interleave(inputs, inputs.shape[0] // batch_size_label)
    logits = engine.model(inputs)
    # logits = de_interleave(logits, inputs.shape[0] // batch_size_label)
    logits_x = logits[:batch_size_label]
    logits_u_w, logits_u_s = logits[batch_size_label:].chunk(2)
    loss_dict_label = engine.train_loss_func(logits_x, targets_x)
    probs_u_w = F.softmax(logits_u_w.detach() / temperture, axis=-1)
    p_targets_u, mask = get_psuedo_label_and_mask(probs_u_w, threshold)
    unlabel_celoss = engine.unlabel_train_loss_func(logits_u_s,
                                                    p_targets_u)["CELoss"]
    unlabel_celoss = (unlabel_celoss * mask).mean()
    loss_dict = dict()
    for k, v in loss_dict_label.items():
        if k != "loss":
            loss_dict[k + "_label"] = v
    loss_dict["CELoss_unlabel"] = unlabel_celoss
    loss_dict["loss"] = loss_dict_label['loss'] + unlabel_celoss
    return loss_dict, logits_x


def get_psuedo_label_and_mask(probs_u_w, threshold):
    max_probs = paddle.max(probs_u_w, axis=-1)
    p_targets_u = paddle.argmax(probs_u_w, axis=-1)

    mask = paddle.greater_equal(max_probs, threshold).astype('float')
    return p_targets_u, mask


def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(
        [1, 0, 2, 3, 4]).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(
        [1, 0, 2]).reshape([-1] + s[1:])
