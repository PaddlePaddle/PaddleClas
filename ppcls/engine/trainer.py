# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import numpy as np
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

import argparse
import paddle
import paddle.nn as nn
import paddle.distributed as dist

from ppcls.utils.check import check_gpu
from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger
from ppcls.data import build_dataloader
from ppcls.arch import build_model
from ppcls.arch.loss_metrics import build_loss
from ppcls.arch.loss_metrics import build_metrics
from ppcls.optimizer import build_optimizer
from ppcls.utils.save_load import load_dygraph_pretrain

from ppcls.utils import save_load


class Trainer(object):
    def __init__(self, config, mode="train"):
        self.mode = mode
        self.config = config
        self.output_dir = self.config['Global']['output_dir']
        # set device
        assert self.config["Global"]["device"] in ["cpu", "gpu", "xpu"]
        self.device = paddle.set_device(self.config["Global"]["device"])
        # set dist
        self.config["Global"][
            "distributed"] = paddle.distributed.get_world_size() != 1
        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
        self.model = build_model(self.config["Arch"])

        if self.config["Global"]["pretrained_model"] is not None:
            load_dygraph_pretrain(self.model,
                                  self.config["Global"]["pretrained_model"])

        if self.config["Global"]["distributed"]:
            self.model = paddle.DataParallel(self.model)

        self.vdl_writer = None
        if self.config['Global']['use_visualdl']:
            from visualdl import LogWriter
            vdl_writer_path = os.path.join(self.output_dir, "vdl")
            if not os.path.exists(vdl_writer_path):
                os.makedirs(vdl_writer_path)
            self.vdl_writer = LogWriter(logdir=vdl_writer_path)
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, self.device))

    def _build_metric_info(self, metric_config, mode="train"):
        """
        _build_metric_info: build metrics according to current mode
        Return:
            metric: dict of the metrics info
        """
        metric = None
        mode = mode.capitalize()
        if mode in metric_config and metric_config[mode] is not None:
            metric = build_metrics(metric_config[mode])
        return metric

    def _build_loss_info(self, loss_config, mode="train"):
        """
        _build_loss_info: build loss according to current mode
        Return:
            loss_dict: dict of the loss info
        """
        loss = None
        mode = mode.capitalize()
        if mode in loss_config and loss_config[mode] is not None:
            loss = build_loss(loss_config[mode])
        return loss

    def train(self):
        # build train loss and metric info
        loss_func = self._build_loss_info(self.config["Loss"])

        metric_func = self._build_metric_info(self.config["Metric"])

        train_dataloader = build_dataloader(self.config["DataLoader"], "train",
                                            self.device)

        step_each_epoch = len(train_dataloader)

        optimizer, lr_sch = build_optimizer(self.config["Optimizer"],
                                            self.config["Global"]["epochs"],
                                            step_each_epoch,
                                            self.model.parameters())

        print_batch_step = self.config['Global']['print_batch_step']
        save_interval = self.config["Global"]["save_interval"]

        best_metric = {
            "metric": 0.0,
            "epoch": 0,
        }
        # key: 
        # val: metrics list word
        output_info = dict()
        # global iter counter
        global_step = 0

        for epoch_id in range(1, self.config["Global"]["epochs"] + 1):
            self.model.train()
            for iter_id, batch in enumerate(train_dataloader()):
                batch_size = batch[0].shape[0]
                batch[1] = paddle.to_tensor(batch[1].numpy().astype("int64")
                                            .reshape([-1, 1]))
                global_step += 1
                # image input
                out = self.model(batch[0])
                # calc loss
                loss_dict = loss_func(out, batch[-1])
                for key in loss_dict:
                    if not key in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(loss_dict[key].numpy()[0],
                                            batch_size)
                # calc metric
                if metric_func is not None:
                    metric_dict = metric_func(out, batch[-1])
                    for key in metric_dict:
                        if not key in output_info:
                            output_info[key] = AverageMeter(key, '7.5f')
                        output_info[key].update(metric_dict[key].numpy()[0],
                                                batch_size)

                if iter_id % print_batch_step == 0:
                    lr_msg = "lr: {:.5f}".format(lr_sch.get_lr())
                    metric_msg = ", ".join([
                        "{}: {:.5f}".format(key, output_info[key].avg)
                        for key in output_info
                    ])
                    logger.info("[Train][Epoch {}][Iter: {}/{}]{}, {}".format(
                        epoch_id, iter_id,
                        len(train_dataloader), lr_msg, metric_msg))

                # step opt and lr
                loss_dict["loss"].backward()
                optimizer.step()
                optimizer.clear_grad()
                lr_sch.step()

            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, output_info[key].avg)
                for key in output_info
            ])
            logger.info("[Train][Epoch {}][Avg]{}".format(epoch_id,
                                                          metric_msg))
            output_info.clear()

            # eval model and save model if possible
            if self.config["Global"][
                    "eval_during_train"] and epoch_id % self.config["Global"][
                        "eval_during_train"] == 0:
                acc = self.eval(epoch_id)
                if acc >= best_metric["metric"]:
                    best_metric["metric"] = acc
                    best_metric["epoch"] = epoch_id
                    save_load.save_model(
                        self.model,
                        optimizer,
                        self.output_dir,
                        model_name=self.config["Arch"]["name"],
                        prefix="best_model")

            # save model
            if epoch_id % save_interval == 0:
                save_load.save_model(
                    self.model,
                    optimizer,
                    self.output_dir,
                    model_name=self.config["Arch"]["name"],
                    prefix="ppcls_epoch_{}".format(epoch_id))

    def build_avg_metrics(self, info_dict):
        return {key: AverageMeter(key, '7.5f') for key in info_dict}

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        output_info = dict()

        eval_dataloader = build_dataloader(self.config["DataLoader"], "eval",
                                           self.device)

        self.model.eval()
        print_batch_step = self.config["Global"]["print_batch_step"]

        # build train loss and metric info
        loss_func = self._build_loss_info(self.config["Loss"], "eval")
        metric_func = self._build_metric_info(self.config["Metric"], "eval")
        metric_key = None

        for iter_id, batch in enumerate(eval_dataloader()):
            batch_size = batch[0].shape[0]
            batch[0] = paddle.to_tensor(batch[0]).astype("float32")
            batch[1] = paddle.to_tensor(batch[1]).reshape([-1, 1])
            # image input
            out = self.model(batch[0])
            # calc build
            if loss_func is not None:
                loss_dict = loss_func(out, batch[-1])
                for key in loss_dict:
                    if not key in output_info:
                        output_info[key] = AverageMeter(key, '7.5f')
                    output_info[key].update(loss_dict[key].numpy()[0],
                                            batch_size)
                # calc metric
                if metric_func is not None:
                    metric_dict = metric_func(out, batch[-1])
                    if paddle.distributed.get_world_size() > 1:
                        for key in metric_dict:
                            paddle.distributed.all_reduce(
                                metric_dict[key],
                                op=paddle.distributed.ReduceOp.SUM)
                            metric_dict[key] = metric_dict[
                                key] / paddle.distributed.get_world_size()
                    for key in metric_dict:
                        if metric_key is None:
                            metric_key = key
                        if not key in output_info:
                            output_info[key] = AverageMeter(key, '7.5f')

                        output_info[key].update(metric_dict[key].numpy()[0],
                                                batch_size)

            if iter_id % print_batch_step == 0:
                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                logger.info("[Eval][Epoch {}][Iter: {}/{}]{}".format(
                    epoch_id, iter_id, len(eval_dataloader), metric_msg))

        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        self.model.train()
        # do not try to save best model
        if metric_func is None:
            return -1
        # return 1st metric in the dict
        return output_info[metric_key].avg