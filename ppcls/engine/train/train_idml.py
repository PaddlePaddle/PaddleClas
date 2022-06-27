# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
import paddle

from ppcls.arch.gears import IDMLNeck
from ppcls.engine.train.utils import log_info, update_loss, update_metric
from ppcls.utils import profiler


def mixup(x, y, alpha):
    batch_size = x.shape[0]    
    lam = np.random.beta(alpha, alpha)                                                                         
    index = paddle.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * paddle.index_select(x, index, axis=0)
    y_a, y_b = y, y[index]                                                               
    return mixed_x, y_a, y_b, lam


def train_epoch_idml(engine, epoch_id, print_batch_step):
    tic = time.time()
    warmup_epoch = engine.config.Global.get('warmup_epoch_idml', -1)
    
    if warmup_epoch > 0 and epoch_id == 1:
        if paddle.distributed.get_world_size() == 1:
            if hasattr(engine.model, 'backbone'):
                for p in engine.model.backbone.parameters():
                    p.stop_gradient = True
            if hasattr(engine.model, 'neck') and isinstance(engine.model.neck, IDMLNeck):
                unfreeze_layer = engine.model.neck.embedding_layer.parameters()
                for p in list(set(engine.model.neck.parameters()).difference(set(unfreeze_layer))):
                    p.stop_gradient = True
            for n, p in engine.model.named_parameters():
                if not p.stop_gradient:
                    print(n)
        else:
            if hasattr(engine.model._layers, 'backbone'):
                for p in engine.model._layers.backbone.parameters():
                    p.stop_gradient = True
            if hasattr(engine.model._layers, 'neck') and isinstance(engine.model._layers.neck, IDMLNeck):
                unfreeze_layer = engine.model._layers.neck.embedding_layer.parameters()
                for p in list(set(engine.model._layers.neck.parameters()).difference(set(unfreeze_layer))):
                    p.stop_gradient = True
    elif warmup_epoch and epoch_id == warmup_epoch + 1:
        if paddle.distributed.get_world_size() == 1:
            for p in engine.model.parameters():
                p.stop_gradient = False
        else:
            for p in engine.model._layers.parameters():
                p.stop_gradient = False

    for iter_id, batch in enumerate(engine.train_dataloader):
        if iter_id >= engine.max_iter:
            break
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        engine.global_step += 1
        x, y = batch[0], batch[1]
        mixed_x, y_1, y_2, lam = mixup(x, y, 1.0)
        out_org = engine.model(x)
        out_mix = engine.model(mixed_x)
        loss_dict = engine.train_loss_func(out_org, y_1)
        loss_mixed1 = engine.train_loss_func(out_mix, y_1)
        loss_mixed2 = engine.train_loss_func(out_mix, y_2)
        
        for k in loss_dict.keys():
            loss_dict[k] = loss_dict[k] + lam * loss_mixed1[k] + \
                (1 - lam) * loss_mixed2[k]

        
        # backward & step opt
        loss_dict["loss"].backward()
                
        for i in range(len(engine.optimizer)):
            engine.optimizer[i].step()

        # clear grad
        for i in range(len(engine.optimizer)):
            engine.optimizer[i].clear_grad()

        # step lr(by step)
        for i in range(len(engine.lr_sch)):
            if not getattr(engine.lr_sch[i], "by_epoch", False):
                engine.lr_sch[i].step()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out_org, batch, batch_size)
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
