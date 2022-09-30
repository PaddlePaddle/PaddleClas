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

import inspect
import time

import paddle
from ppcls.engine.train.utils import update_loss, update_metric, log_info
from ppcls.utils import profiler
from ppcls.utils import save_load
from ppcls.utils import logger


def train_epoch(engine, epoch_id, print_batch_step):
    tic = time.time()
    for iter_id, batch in enumerate(engine.train_dataloader):
        if iter_id >= engine.max_iter:
            break
        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]["data"]),
                paddle.to_tensor(batch[0]["label"])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([batch_size, -1])
        engine.global_step += 1

        # image input
        if engine.amp:
            amp_level = engine.config["AMP"].get("level", "O1").upper()
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=amp_level):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])
        else:
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])

        # loss
        loss = loss_dict["loss"] / engine.update_freq

        # backward & step opt
        if engine.amp:
            scaled = engine.scaler.scale(loss)
            scaled.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.scaler.minimize(engine.optimizer[i], scaled)
        else:
            loss.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.optimizer[i].step()

        if (iter_id + 1) % engine.update_freq == 0:
            # clear grad
            for i in range(len(engine.optimizer)):
                engine.optimizer[i].clear_grad()
            # step lr(by step)
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    engine.lr_sch[i].step()
            # update ema
            if engine.ema:
                engine.model_ema.update(engine.model)

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
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


def train_iter(engine, epoch_id, print_batch_step, best_metric):
    """
    train by iteration, epoch_id is fixed to 1
    """
    assert epoch_id == 1, f"epoch_id({epoch_id}) must equal to 1 in train_iter function"
    tic = time.time()

    save_interval = engine.config["Global"]["save_interval"]
    ema_module = None
    if engine.ema:
        best_metric_ema = 0.0
        ema_module = engine.model_ema.module

    dataloader_iterator = iter(engine.train_dataloader)
    for iter_id in range(best_metric.get("iters", -1) + 1, engine.max_iter):
        # fetch data from dataloader iterator
        batch = dataloader_iterator.next()

        profiler.add_profiler_step(engine.config["profiler_options"])
        if iter_id == 5:
            for key in engine.time_info:
                engine.time_info[key].reset()
        engine.time_info["reader_cost"].update(time.time() - tic)

        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]["data"]),
                paddle.to_tensor(batch[0]["label"])
            ]
        batch_size = batch[0].shape[0]
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([batch_size, -1])
        engine.global_step += 1

        # image input
        if engine.amp:
            amp_level = engine.config["AMP"].get("level", "O1").upper()
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=amp_level):
                out = forward(engine, batch)
                loss_dict = engine.train_loss_func(out, batch[1])
        else:
            out = forward(engine, batch)
            loss_dict = engine.train_loss_func(out, batch[1])

        # loss
        loss = loss_dict["loss"] / engine.update_freq

        # backward & step opt
        if engine.amp:
            scaled = engine.scaler.scale(loss)
            scaled.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.scaler.minimize(engine.optimizer[i], scaled)
        else:
            loss.backward()
            if (iter_id + 1) % engine.update_freq == 0:
                for i in range(len(engine.optimizer)):
                    engine.optimizer[i].step()

        if (iter_id + 1) % engine.update_freq == 0:
            # clear grad
            for i in range(len(engine.optimizer)):
                engine.optimizer[i].clear_grad()
            # step lr(by step)
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    argspec = inspect.getargspec(engine.lr_sch[i].step).args
                    if "metrics" not in argspec:
                        engine.lr_sch[i].step()

            # update ema
            if engine.ema:
                engine.model_ema.update(engine.model)

        start_eval_iter = engine.config["Global"].get("start_eval_iter", 0) - 1
        if engine.config["Global"][
                "eval_during_train"] and iter_id % engine.config["Global"][
                    "eval_interval"] == 0 and iter_id > start_eval_iter:
            acc = engine.eval(epoch_id)
            if acc > best_metric["metric"]:
                best_metric["metric"] = acc
                best_metric["iter"] = iter_id
                save_load.save_model(
                    engine.model,
                    engine.optimizer,
                    best_metric,
                    engine.output_dir,
                    ema=ema_module,
                    model_name=engine.config["Arch"]["name"],
                    prefix="best_model",
                    loss=engine.train_loss_func,
                    save_student_model=True)
            logger.info("[Eval][Iter {}][best metric: {}]".format(
                iter_id, best_metric["metric"]))
            logger.scaler(
                name="eval_acc",
                value=acc,
                step=iter_id,
                writer=engine.vdl_writer)

            # step lr by metric, such as `ReduceOnPlateau`
            for i in range(len(engine.lr_sch)):
                if not getattr(engine.lr_sch[i], "by_epoch", False):
                    argspec = inspect.getargspec(engine.lr_sch[i].step).args
                    if "metrics" in argspec:
                        engine.lr_sch[i].step(metrics=acc)

            engine.model.train()

            if engine.ema:
                ori_model, engine.model = engine.model, ema_module
                acc_ema = engine.eval(epoch_id)
                engine.model = ori_model
                ema_module.eval()

                if acc_ema > best_metric_ema:
                    best_metric_ema = acc_ema
                    save_load.save_model(
                        engine.model,
                        engine.optimizer,
                        {"metric": acc_ema,
                         "iter": iter_id},
                        engine.output_dir,
                        ema=ema_module,
                        model_name=engine.config["Arch"]["name"],
                        prefix="best_model_ema",
                        loss=engine.train_loss_func)
                logger.info("[Eval][Iter {}][best metric ema: {}]".format(
                    iter_id, best_metric_ema))
                logger.scaler(
                    name="eval_acc_ema",
                    value=acc_ema,
                    step=iter_id,
                    writer=engine.vdl_writer)

            # save model
            if iter_id > 0 and iter_id % save_interval == 0:
                save_load.save_model(
                    engine.model,
                    engine.optimizer, {"metric": acc,
                                       "iter": iter_id},
                    engine.output_dir,
                    ema=ema_module,
                    model_name=engine.config["Arch"]["name"],
                    prefix="iter_{}".format(iter_id),
                    loss=engine.train_loss_func)

        if engine.vdl_writer is not None:
            engine.vdl_writer.close()

        # below code just for logging
        # update metric_for_logger
        update_metric(engine, out, batch, batch_size)
        # update_loss_for_logger
        update_loss(engine, loss_dict, batch_size)
        engine.time_info["batch_cost"].update(time.time() - tic)
        if iter_id % print_batch_step == 0:
            log_info(engine, batch_size, epoch_id, iter_id)
        tic = time.time()


def forward(engine, batch):
    if not engine.is_rec:
        return engine.model(batch[0])
    else:
        return engine.model(batch[0], batch[1])
