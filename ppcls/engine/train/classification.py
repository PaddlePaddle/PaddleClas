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

from .utils import update_loss, update_metric, log_info
from ...utils import logger, profiler, type_name
from ...utils.misc import AverageMeter
from ...data import build_dataloader
from ...loss import build_loss
from ...metric import build_metrics
from ...optimizer import build_optimizer
from ...utils.ema import ExponentialMovingAverage
from ...utils.save_load import init_model, ModelSaver


class ClassTrainer(object):
    def __init__(self, config, model, eval_func):
        self.config = config
        self.model = model
        self.eval = eval_func
        self.start_eval_epoch = self.config["Global"].get("start_eval_epoch",
                                                          0) - 1
        self.epochs = self.config["Global"].get("epochs", 1)
        self.print_batch_step = self.config['Global']['print_batch_step']
        self.save_interval = self.config["Global"]["save_interval"]
        self.output_dir = self.config['Global']['output_dir']
        # gradient accumulation
        self.update_freq = self.config["Global"].get("update_freq", 1)

        if "Head" in self.config["Arch"] or self.config["Arch"].get("is_rec",
                                                                    False):
            self.is_rec = True
        else:
            self.is_rec = False

        # TODO(gaotingquan): mv to build_model
        # build EMA model
        self.model_ema = self._build_ema_model()

        # build dataloader
        self.use_dali = self.config["Global"].get("use_dali", False)
        self.dataloader = build_dataloader(self.config, "Train")
        self.dataloader_iter = iter(self.dataloader)

        # build loss
        self.loss_func = build_loss(config, "Train")

        # build metric
        self.train_metric_func = build_metrics(config, "Train")

        # build optimizer
        self.optimizer, self.lr_sch = build_optimizer(
            self.config, self.dataloader.max_iter,
            [self.model, self.loss_func])

        # build model saver
        self.model_saver = ModelSaver(
            config=self.config,
            net=self.model,
            loss=self.loss_func,
            opt=self.optimizer,
            model_ema=self.model_ema if self.model_ema else None)

        # build best metric
        self.best_metric = {
            "metric": -1.0,
            "epoch": 0,
        }

        # key:
        # val: metrics list word
        self.output_info = dict()
        self.time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }

        self._init_checkpoints()

        # for visualdl
        self.vdl_writer = self._init_vdl()

    def __call__(self):
        # global iter counter
        self.global_step = 0
        for epoch_id in range(self.best_metric["epoch"] + 1, self.epochs + 1):
            # for one epoch train
            self.train_epoch(epoch_id)

            metric_msg = ", ".join(
                [self.output_info[key].avg_info for key in self.output_info])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.epochs, metric_msg))
            self.output_info.clear()

            acc = 0.0
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_interval"] == 0 and epoch_id > self.start_eval_epoch:
                acc = self.eval(epoch_id)

                # step lr (by epoch) according to given metric, such as acc
                for i in range(len(self.lr_sch)):
                    if getattr(self.lr_sch[i], "by_epoch", False) and \
                            type_name(self.lr_sch[i]) == "ReduceOnPlateau":
                        self.lr_sch[i].step(acc)

                if acc > self.best_metric["metric"]:
                    self.best_metric["metric"] = acc
                    self.best_metric["epoch"] = epoch_id
                    self.model_saver.save(
                        self.best_metric,
                        prefix="best_model",
                        save_student_model=True)

                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, self.best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=acc,
                    step=epoch_id,
                    writer=self.vdl_writer)

                self.model.train()

                if self.model_ema:
                    ori_model, self.model = self.model, self.model_ema.module
                    acc_ema = self.eval(epoch_id)
                    self.model = ori_model
                    self.model_ema.module.eval()

                    if acc_ema > self.best_metric["metric_ema"]:
                        self.best_metric["metric_ema"] = acc_ema
                        self.model_saver.save(
                            {
                                "metric": acc_ema,
                                "epoch": epoch_id
                            },
                            prefix="best_model_ema")
                    logger.info("[Eval][Epoch {}][best metric ema: {}]".format(
                        epoch_id, self.best_metric["metric_ema"]))
                    logger.scaler(
                        name="eval_acc_ema",
                        value=acc_ema,
                        step=epoch_id,
                        writer=self.vdl_writer)

            # save model
            if self.save_interval > 0 and epoch_id % self.save_interval == 0:
                self.model_saver.save(
                    {
                        "metric": acc,
                        "epoch": epoch_id
                    },
                    prefix=f"epoch_{epoch_id}")

            # save the latest model
            self.model_saver.save(
                {
                    "metric": acc,
                    "epoch": epoch_id
                }, prefix="latest")

    def train_epoch(self, epoch_id):
        self.model.train()
        tic = time.time()

        for iter_id in range(self.dataloader.max_iter):
            # fetch data batch from dataloader
            try:
                batch = next(self.dataloader_iter)
            except Exception:
                # NOTE: reset DALI dataloader manually
                if self.use_dali:
                    self.dataloader.reset()
                self.dataloader_iter = iter(self.dataloader)
                batch = next(self.dataloader_iter)

            profiler.add_profiler_step(self.config["profiler_options"])
            if iter_id == 5:
                for key in self.time_info:
                    self.time_info[key].reset()
            self.time_info["reader_cost"].update(time.time() - tic)

            batch_size = batch[0].shape[0]
            if not self.config["Global"].get("use_multilabel", False):
                batch[1] = batch[1].reshape([batch_size, -1])
            self.global_step += 1

            if self.is_rec:
                out = self.model(batch)
            else:
                out = self.model(batch[0])

            loss_dict = self.loss_func(out, batch[1])
            # TODO(gaotingquan): mv update_freq to loss and optimizer
            loss = loss_dict["loss"] / self.update_freq
            loss.backward()

            if (iter_id + 1) % self.update_freq == 0:
                for i in range(len(self.optimizer)):
                    self.optimizer[i].step()

            if (iter_id + 1) % self.update_freq == 0:
                # clear grad
                for i in range(len(self.optimizer)):
                    self.optimizer[i].clear_grad()
                # step lr(by step)
                for i in range(len(self.lr_sch)):
                    if not getattr(self.lr_sch[i], "by_epoch", False):
                        self.lr_sch[i].step()
                # update ema
                if self.model_ema:
                    self.model_ema.update(self.model)

            # below code just for logging
            # update metric_for_logger
            update_metric(self, out, batch, batch_size)
            # update_loss_for_logger
            update_loss(self, loss_dict, batch_size)
            self.time_info["batch_cost"].update(time.time() - tic)
            if iter_id % self.print_batch_step == 0:
                log_info(self, batch_size, epoch_id, iter_id)
            tic = time.time()

        # step lr(by epoch)
        for i in range(len(self.lr_sch)):
            if getattr(self.lr_sch[i], "by_epoch", False) and \
                    type_name(self.lr_sch[i]) != "ReduceOnPlateau":
                self.lr_sch[i].step()

    def __del__(self):
        if self.vdl_writer is not None:
            self.vdl_writer.close()

    def _init_vdl(self):
        if self.config['Global']['use_visualdl'] and dist.get_rank() == 0:
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            return LogWriter(logdir=vdl_writer_path)
        return None

    def _build_ema_model(self):
        if "EMA" in self.config:
            model_ema = ExponentialMovingAverage(
                self.model, self.config['EMA'].get("decay", 0.9999))
            self.best_metric["metric_ema"] = 0
            return model_ema
        else:
            return None

    def _init_checkpoints(self):
        if self.config["Global"].get("checkpoints", None) is not None:
            metric_info = init_model(self.config.Global, self.model,
                                     self.optimizer, self.train_loss_func,
                                     self.model_ema)
            if metric_info is not None:
                self.best_metric.update(metric_info)