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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle

from ...utils.misc import AverageMeter
from ...utils import logger
from ...data import build_dataloader
from ...loss import build_loss
from ...metric import build_metrics


class ClassEval(object):
    def __init__(self, config, mode, model):
        self.config = config
        self.model = model
        self.print_batch_step = self.config["Global"]["print_batch_step"]
        self.use_dali = self.config["Global"].get("use_dali", False)
        self.eval_metric_func = build_metrics(self.config, "Eval")
        self.eval_dataloader = build_dataloader(self.config, "Eval")
        self.eval_loss_func = build_loss(self.config, "Eval")
        self.output_info = dict()

    @paddle.no_grad()
    def __call__(self, epoch_id=0):
        self.model.eval()

        if hasattr(self.eval_metric_func, "reset"):
            self.eval_metric_func.reset()

        time_info = {
            "batch_cost": AverageMeter(
                "batch_cost", '.5f', postfix=" s,"),
            "reader_cost": AverageMeter(
                "reader_cost", ".5f", postfix=" s,"),
        }

        tic = time.time()
        total_samples = self.eval_dataloader.total_samples
        accum_samples = 0
        max_iter = self.eval_dataloader.max_iter
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
                loss_dict = self.eval_loss_func(preds, labels)

                for key in loss_dict:
                    if key not in self.output_info:
                        self.output_info[key] = AverageMeter(key, '7.5f')
                    self.output_info[key].update(
                        float(loss_dict[key]), current_samples)

            #  calc metric
            if self.eval_metric_func is not None:
                self.eval_metric_func(preds, labels)
            time_info["batch_cost"].update(time.time() - tic)

            if iter_id % self.print_batch_step == 0:
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
                        "{}: {:.5f}".format(key, self.output_info[key].val)
                        for key in self.output_info
                    ])
                    metric_msg += ", {}".format(self.eval_metric_func.avg_info)
                logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                    epoch_id, iter_id, max_iter, metric_msg, time_msg,
                    ips_msg))

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
                "{}: {:.5f}".format(key, self.output_info[key].avg)
                for key in self.output_info
            ])
            metric_msg += ", {}".format(self.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

            # do not try to save best eval.model
            if self.eval_metric_func is None:
                return -1
            # return 1st metric in the dict
            return self.eval_metric_func.avg
        self.model.train()
        return eval_result
