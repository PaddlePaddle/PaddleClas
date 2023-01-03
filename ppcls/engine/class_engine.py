# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
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

import os
import time
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random
from ppcls.engine.base_engine import BaseEngine
from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.data import build_dataloader
from ppcls.arch import build_model, RecModel, DistillationModel, TheseusLayer
from ppcls.arch import apply_to_static
from ppcls.loss import build_loss
from ppcls.metric import build_metrics
from ppcls.optimizer import build_optimizer
from ppcls.utils.ema import ExponentialMovingAverage
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url
from ppcls.utils.save_load import init_model
from ppcls.utils import save_load

from ppcls.data.utils.get_image_list import get_image_list
from ppcls.data.postprocess import build_postprocess
from ppcls.data import create_operators
from ppcls.self import train as train_method
from ppcls.self.train.utils import type_name
from ppcls.arch.gears.identity_head import IdentityHead
from ppcls.self.train.utils import update_loss, update_metric, log_info, type_name
from ppcls.utils import profiler


class ClassEngine(BaseEngine):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode=mode)
        self._build_component()
        self._set_train_attribute()

    def train_epoch(self, epoch_id, print_batch_step):
        tic = time.time()
        if not hasattr(self, "train_dataloader_iter"):
            self.train_dataloader_iter = iter(self.train_dataloader)

        for iter_id in range(self.iter_per_epoch):
            # fetch data batch from dataloader
            try:
                batch = next(self.train_dataloader_iter)
            except Exception:
                self.train_dataloader_iter = iter(self.train_dataloader)
                batch = next(self.train_dataloader_iter)

            profiler.add_profiler_step(self.config["profiler_options"])
            if iter_id == 5:
                for key in self.time_info:
                    self.time_info[key].reset()
            self.time_info["reader_cost"].update(time.time() - tic)

            batch_size = batch[0].shape[0]
            if not self.config["Global"].get("use_multilabel", False):
                batch[1] = batch[1].reshape([batch_size, -1])
            self.global_step += 1

            # image input
            if self.amp:
                amp_level = self.config["AMP"].get("level", "O1").upper()
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=amp_level):
                    out = self.model(batch[0])
                    loss_dict = self.train_loss_func(out, batch[1])
            else:
                out = self.model(batch[0])
                loss_dict = self.train_loss_func(out, batch[1])

            # loss
            loss = loss_dict["loss"] / self.update_freq

            # backward & step opt
            if self.amp:
                scaled = self.scaler.scale(loss)
                scaled.backward()
                if (iter_id + 1) % self.update_freq == 0:
                    for i in range(len(self.optimizer)):
                        self.scaler.minimize(self.optimizer[i], scaled)
            else:
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
                if self.ema:
                    self.model_ema.update(self.model)

            # below code just for logging
            # update metric_for_logger
            update_metric(self, out, batch, batch_size)
            # update_loss_for_logger
            update_loss(self, loss_dict, batch_size)
            self.time_info["batch_cost"].update(time.time() - tic)
            if iter_id % print_batch_step == 0:
                log_info(self, batch_size, epoch_id, iter_id)
            tic = time.time()

        # step lr(by epoch)
        for i in range(len(self.lr_sch)):
            if getattr(self.lr_sch[i], "by_epoch", False) and \
                    type_name(self.lr_sch[i]) != "ReduceOnPlateau":
                self.lr_sch[i].step()

    def eval_epoch(self, epoch_id):
        if hasattr(self.eval_metric_func, "reset"):
            self.eval_metric_func.reset()
        output_info = dict()
        time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }
        print_batch_step = self.config["Global"]["print_batch_step"]

        tic = time.time()
        accum_samples = 0
        total_samples = len(
            self.eval_dataloader.
            dataset) if not self.use_dali else self.eval_dataloader.size
        max_iter = len(self.eval_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.eval_dataloader)
        for iter_id, batch in enumerate(self.eval_dataloader):
            if iter_id >= max_iter:
                break
            if iter_id == 5:
                for key in time_info:
                    time_info[key].reset()

            time_info["reader_cost"].update(time.time() - tic)
            batch_size = batch[0].shape[0]
            batch[0] = paddle.to_tensor(batch[0])
            if not self.config["Global"].get("use_multilabel", False):
                batch[1] = batch[1].reshape([-1, 1]).astype("int64")

            # image input
            if self.amp and self.amp_eval:
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=self.amp_level):
                    out = self.model(batch[0])
            else:
                out = self.model(batch[0])

            # just for DistributedBatchSampler issue: repeat sampling
            current_samples = batch_size * paddle.distributed.get_world_size()
            accum_samples += current_samples

            if isinstance(out, dict) and "Student" in out:
                out = out["Student"]
            if isinstance(out, dict) and "logits" in out:
                out = out["logits"]

            # gather Tensor when distributed
            if paddle.distributed.get_world_size() > 1:
                label_list = []
                device_id = paddle.distributed.ParallelEnv().device_id
                label = batch[1].cuda(device_id) if self.config["Global"][
                    "device"] == "gpu" else batch[1]
                paddle.distributed.all_gather(label_list, label)
                labels = paddle.concat(label_list, 0)

                if isinstance(out, list):
                    preds = []
                    for x in out:
                        pred_list = []
                        paddle.distributed.all_gather(pred_list, x)
                        pred_x = paddle.concat(pred_list, 0)
                        preds.append(pred_x)
                else:
                    pred_list = []
                    paddle.distributed.all_gather(pred_list, out)
                    preds = paddle.concat(pred_list, 0)

                if accum_samples > total_samples and not self.use_dali:
                    if isinstance(preds, list):
                        preds = [
                            pred[:total_samples + current_samples -
                                 accum_samples] for pred in preds
                        ]
                    else:
                        preds = preds[:total_samples + current_samples -
                                      accum_samples]
                    labels = labels[:total_samples + current_samples -
                                    accum_samples]
                    current_samples = total_samples + current_samples - accum_samples
            else:
                labels = batch[1]
                preds = out

            # calc loss
            if self.eval_loss_func is not None:
                if self.amp and self.amp_eval:
                    with paddle.amp.auto_cast(
                            custom_black_list={
                                "flatten_contiguous_range", "greater_than"
                            },
                            level=self.amp_level):
                        loss_dict = self.eval_loss_func(preds, labels)
                else:
                    loss_dict = self.eval_loss_func(preds, labels)

                for key in loss_dict:
                    if key not in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(
                        float(loss_dict[key]), current_samples)

            #  calc metric
            if self.eval_metric_func is not None:
                self.eval_metric_func(preds, labels)
            time_info["batch_cost"].update(time.time() - tic)

            if iter_id % print_batch_step == 0:
                time_msg = "s, ".join([
                    "{}: {:.5f}".format(key, time_info[key].avg)
                    for key in time_info
                ])

                ips_msg = "ips: {:.5f} images/sec".format(
                    batch_size / time_info["batch_cost"].avg)

                if "ATTRMetric" in self.config["Metric"]["Eval"][0]:
                    metric_msg = ""
                else:
                    metric_msg = ", ".join([
                        "{}: {:.5f}".format(key, output_info[key].val)
                        for key in output_info
                    ])
                    metric_msg += ", {}".format(self.eval_metric_func.avg_info)
                logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                    epoch_id, iter_id,
                    len(self.eval_dataloader), metric_msg, time_msg, ips_msg))

            tic = time.time()
        if self.use_dali:
            self.eval_dataloader.reset()

        if "ATTRMetric" in self.config["Metric"]["Eval"][0]:
            metric_msg = ", ".join([
                "evalres: ma: {:.5f} label_f1: {:.5f} label_pos_recall: {:.5f} label_neg_recall: {:.5f} instance_f1: {:.5f} instance_acc: {:.5f} instance_prec: {:.5f} instance_recall: {:.5f}".
                format(*self.eval_metric_func.attr_res())
            ])
            logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

            # do not try to save best eval.model
            if self.eval_metric_func is None:
                return -1
            # return 1st metric in the dict
            return self.eval_metric_func.attr_res()[0]
        else:
            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].avg)
                for key in output_info
            ])
            metric_msg += ", {}".format(self.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

            # do not try to save best eval.model
            if self.eval_metric_func is None:
                return -1
            # return 1st metric in the dict
            return self.eval_metric_func.avg

    @paddle.no_grad()
    def infer(self):
        assert self.mode == "infer" and self.eval_mode == "classification"
        total_trainer = dist.get_world_size()
        local_rank = dist.get_rank()
        image_list = get_image_list(self.config["Infer"]["infer_imgs"])
        # data split
        image_list = image_list[local_rank::total_trainer]

        batch_size = self.config["Infer"]["batch_size"]
        self.model.eval()
        batch_data = []
        image_file_list = []
        for idx, image_file in enumerate(image_list):
            with open(image_file, 'rb') as f:
                x = f.read()
            for process in self.preprocess_func:
                x = process(x)
            batch_data.append(x)
            image_file_list.append(image_file)
            if len(batch_data) >= batch_size or idx == len(image_list) - 1:
                batch_tensor = paddle.to_tensor(batch_data)

                if self.amp and self.amp_eval:
                    with paddle.amp.auto_cast(
                            custom_black_list={
                                "flatten_contiguous_range", "greater_than"
                            },
                            level=self.amp_level):
                        out = self.model(batch_tensor)
                else:
                    out = self.model(batch_tensor)

                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, dict) and "Student" in out:
                    out = out["Student"]
                if isinstance(out, dict) and "logits" in out:
                    out = out["logits"]
                if isinstance(out, dict) and "output" in out:
                    out = out["output"]
                result = self.postprocess_func(out, image_file_list)
                print(result)
                batch_data.clear()
                image_file_list.clear()
