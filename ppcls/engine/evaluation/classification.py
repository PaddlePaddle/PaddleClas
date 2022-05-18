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

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def classification_eval(engine, epoch_id=0):
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0])
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        # image input
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0])
        else:
            out = engine.model(batch[0])

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

            paddle.distributed.all_gather(label_list, batch[1])
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

            if accum_samples > total_samples and not engine.use_dali:
                preds = preds[:total_samples + current_samples - accum_samples]
                labels = labels[:total_samples + current_samples -
                                accum_samples]
                current_samples = total_samples + current_samples - accum_samples
        else:
            labels = batch[1]
            preds = out

        # calc loss
        if engine.eval_loss_func is not None:
            if engine.amp and engine.amp_eval:
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=engine.amp_level):
                    loss_dict = engine.eval_loss_func(preds, labels)
            else:
                loss_dict = engine.eval_loss_func(preds, labels)

            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0],
                                        current_samples)

        #  calc metric
        if engine.eval_metric_func is not None:
            engine.eval_metric_func(preds, labels)
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
                metric_msg = ""
            else:
                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()

    if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
        metric_msg = ", ".join([
            "evalres: ma: {:.5f} label_f1: {:.5f} label_pos_recall: {:.5f} label_neg_recall: {:.5f} instance_f1: {:.5f} instance_acc: {:.5f} instance_prec: {:.5f} instance_recall: {:.5f}".
            format(*engine.eval_metric_func.attr_res())
        ])
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.attr_res()[0]
    else:
        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.avg
