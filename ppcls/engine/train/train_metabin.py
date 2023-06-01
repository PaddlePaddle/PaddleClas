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

# reference: https://arxiv.org/abs/2011.14670v2

from __future__ import absolute_import, division, print_function

import time
import paddle
import numpy as np
from collections import defaultdict

from ppcls.engine.train.utils import update_loss, update_metric, log_info, type_name
from ppcls.utils import profiler
from ppcls.data import build_dataloader
from ppcls.loss import build_loss


def train_epoch_metabin(engine, epoch_id, print_batch_step):
    tic = time.time()

    if not hasattr(engine, "train_dataloader_iter"):
        engine.train_dataloader_iter = iter(engine.train_dataloader)

    if not hasattr(engine, "meta_dataloader"):
        engine.meta_dataloader = build_dataloader(
            config=engine.config['DataLoader']['Metalearning'],
            mode='Train',
            device=engine.device)
        engine.meta_dataloader_iter = iter(engine.meta_dataloader)

    num_domain = engine.train_dataloader.dataset.num_cams
    for iter_id in range(engine.iter_per_epoch):
        # fetch data batch from dataloader
        try:
            train_batch = next(engine.train_dataloader_iter)
        except Exception:
            engine.train_dataloader_iter = iter(engine.train_dataloader)
            train_batch = next(engine.train_dataloader_iter)

        try:
            mtrain_batch, mtest_batch = get_meta_data(
                engine.meta_dataloader_iter, num_domain)
        except Exception:
            engine.meta_dataloader_iter = iter(engine.meta_dataloader)
            mtrain_batch, mtest_batch = get_meta_data(
                engine.meta_dataloader_iter, num_domain)

        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        train_batch_size = train_batch[0].shape[0]
        mtrain_batch_size = mtrain_batch[0].shape[0]
        mtest_batch_size = mtest_batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            train_batch[1] = train_batch[1].reshape([train_batch_size, -1])
            mtrain_batch[1] = mtrain_batch[1].reshape([mtrain_batch_size, -1])
            mtest_batch[1] = mtest_batch[1].reshape([mtest_batch_size, -1])

        engine.global_step += 1

        if engine.global_step == 1:  # update model (execpt gate) to warmup
            for i in range(engine.config["Global"]["warmup_iter"] - 1):
                out, basic_loss_dict = basic_update(engine, train_batch)
                loss_dict = basic_loss_dict
                try:
                    train_batch = next(engine.train_dataloader_iter)
                except Exception:
                    engine.train_dataloader_iter = iter(
                        engine.train_dataloader)
                    train_batch = next(engine.train_dataloader_iter)

        out, basic_loss_dict = basic_update(engine=engine, batch=train_batch)
        mtrain_loss_dict, mtest_loss_dict = metalearning_update(
            engine=engine, mtrain_batch=mtrain_batch, mtest_batch=mtest_batch)
        loss_dict = {
            **
            {"train_" + key: value
             for key, value in basic_loss_dict.items()}, ** {
                 "mtrain_" + key: value
                 for key, value in mtrain_loss_dict.items()
             }, **
            {"mtest_" + key: value
             for key, value in mtest_loss_dict.items()}
        }
        # step lr (by iter)
        for i in range(len(engine.lr_sch)):
            if not getattr(engine.lr_sch[i], "by_epoch", False):
                engine.lr_sch[i].step()
        # update ema
        if engine.ema:
            engine.model_ema.update(engine.model)

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, train_batch, train_batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, train_batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, train_batch_size, epoch_id, iter_id)
        tic = time.time()

    # step lr(by epoch)
    for i in range(len(engine.lr_sch)):
        if getattr(engine.lr_sch[i], "by_epoch", False) and \
                type_name(engine.lr_sch[i]) != "ReduceOnPlateau":
            engine.lr_sch[i].step()


def setup_opt(engine, stage):
    assert stage in ["train", "mtrain", "mtest"]
    opt = defaultdict()
    if stage == "train":
        opt["bn_mode"] = "general"
        opt["enable_inside_update"] = False
        opt["lr_gate"] = 0.0
    elif stage == "mtrain":
        opt["bn_mode"] = "hold"
        opt["enable_inside_update"] = False
        opt["lr_gate"] = 0.0
    elif stage == "mtest":
        norm_lr = engine.lr_sch[1].last_lr
        cyclic_lr = engine.lr_sch[2].get_lr()
        opt["bn_mode"] = "hold"
        opt["enable_inside_update"] = True
        opt["lr_gate"] = norm_lr * cyclic_lr
    for layer in engine.model.backbone.sublayers():
        if type_name(layer) == "MetaBIN":
            layer.setup_opt(opt)
    engine.model.neck.setup_opt(opt)


def reset_opt(model):
    for layer in model.backbone.sublayers():
        if type_name(layer) == "MetaBIN":
            layer.reset_opt()
    model.neck.reset_opt()


def get_meta_data(meta_dataloader_iter, num_domain):
    """
    fetch data batch from dataloader then divide the batch by domains
    """
    list_all = np.random.permutation(num_domain)
    list_mtrain = list(list_all[:num_domain // 2])
    batch = next(meta_dataloader_iter)
    domain_idx = batch[2]
    cnt = 0
    for sample in list_mtrain:
        if cnt == 0:
            is_mtrain_domain = domain_idx == sample
        else:
            is_mtrain_domain = paddle.logical_or(is_mtrain_domain,
                                                 domain_idx == sample)
        cnt += 1

    # mtrain_batch
    if not any(is_mtrain_domain):
        mtrain_batch = None
        raise RuntimeError
    else:
        mtrain_batch = [batch[i][is_mtrain_domain] for i in range(len(batch))]

    # mtest_batch
    is_mtest_domains = is_mtrain_domain == False
    if not any(is_mtest_domains):
        mtest_batch = None
        raise RuntimeError
    else:
        mtest_batch = [batch[i][is_mtest_domains] for i in range(len(batch))]
    return mtrain_batch, mtest_batch


def forward(engine, batch, loss_func):
    batch_info = defaultdict()
    batch_info = {"label": batch[1], "domain": batch[2]}

    with engine.auto_cast(is_eval=False):
        out = engine.model(batch[0], batch[1])
        loss_dict = loss_func(out, batch_info)

    return out, loss_dict


def backward(engine, loss, optimizer):
    optimizer.clear_grad()
    scaled = engine.scaler.scale(loss)
    scaled.backward()

    # optimizer.step() with auto amp
    engine.scaler.step(optimizer)
    engine.scaler.update()

    for name, layer in engine.model.backbone.named_sublayers():
        if "gate" == name.split('.')[-1]:
            layer.clip_gate()


def basic_update(engine, batch):
    setup_opt(engine, "train")
    train_loss_func = build_loss(engine.config["Loss"]["Basic"])
    out, train_loss_dict = forward(engine, batch, train_loss_func)
    train_loss = train_loss_dict["loss"]
    backward(engine, train_loss, engine.optimizer[0])
    engine.optimizer[0].clear_grad()
    reset_opt(engine.model)
    return out, train_loss_dict


def metalearning_update(engine, mtrain_batch, mtest_batch):
    # meta train
    mtrain_loss_func = build_loss(engine.config["Loss"]["MetaTrain"])
    setup_opt(engine, "mtrain")

    mtrain_batch_info = defaultdict()
    mtrain_batch_info = {"label": mtrain_batch[1], "domain": mtrain_batch[2]}
    out = engine.model(mtrain_batch[0], mtrain_batch[1])
    mtrain_loss_dict = mtrain_loss_func(out, mtrain_batch_info)
    mtrain_loss = mtrain_loss_dict["loss"]
    engine.optimizer[1].clear_grad()
    mtrain_loss.backward()

    # meta test
    mtest_loss_func = build_loss(engine.config["Loss"]["MetaTest"])
    setup_opt(engine, "mtest")

    out, mtest_loss_dict = forward(engine, mtest_batch, mtest_loss_func)
    engine.optimizer[1].clear_grad()
    mtest_loss = mtest_loss_dict["loss"]
    backward(engine, mtest_loss, engine.optimizer[1])

    engine.optimizer[0].clear_grad()
    engine.optimizer[1].clear_grad()
    reset_opt(engine.model)

    return mtrain_loss_dict, mtest_loss_dict
