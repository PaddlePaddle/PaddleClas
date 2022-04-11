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
import platform
from typing import Callable

import paddle
import paddle.distributed as dist
from paddle import nn
from ppcls.arch import (DistillationModel, RecModel, TheseusLayer)
from ppcls.arch.gears.identity_head import IdentityHead
from ppcls.data import create_operators
from ppcls.data.postprocess import build_postprocess
from ppcls.data.utils.get_image_list import get_image_list
from ppcls.engine import evaluation
from ppcls.engine.train import train_epoch
from ppcls.utils import (load_pretrain, logger, save_load, set_amp,
                         set_dataloaders, set_device, set_distributed,
                         set_logger, set_losses, set_metrics, set_model,
                         set_optimizer, set_seed, set_visualDL)
from ppcls.utils.check import check_gpu
from ppcls.utils.config import AttrDict
from ppcls.utils.env_init import set_amp
from ppcls.utils.misc import AverageMeter
from ppcls.utils.save_load import init_model, load_dygraph_pretrain


class Engine(object):
    def __init__(self, config: AttrDict, mode: str="train"):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode = mode
        self.config = config
        self.eval_mode = self.config.Global.get("eval_mode", "classification")
        if "Head" in self.config.Arch or self.config.Arch.get("is_rec", False):
            self.is_rec = True
        else:
            self.is_rec = False

        ## 1 environment
        # 1.1  seed if specified
        seed = self.config.Global.get("seed", None)
        set_seed(seed)

        # 1.2 init logger
        set_logger(self, print_cfg=True)

        # 1.3 set visualDL
        set_visualDL(self)

        # 1.4 set device
        set_device(self)

        # 1.5 set train_func and eval_func
        assert self.eval_mode in ["classification", "retrieval"], logger.error(
            "Invalid eval mode: {}".format(self.eval_mode))
        self.train_epoch_func: Callable = train_epoch
        self.eval_func: Callable = getattr(evaluation,
                                           self.eval_mode + "_eval")

        ## 2 dataloader(s)
        set_dataloaders(self)

        ## 3 losses
        set_losses(self)

        ## 4 build model(s)
        # 4.1 initialize model(s)
        set_model(self)

        # 4.2 load_pretrain
        if self.config.Global.pretrained_model is not None:
            load_pretrain(self)

        ## 5 optimizer
        set_optimizer(self)

        ## 6 amp configurations
        set_amp(self)

        ## 7 distributed
        set_distributed(self)

        ## 8 metric(s)
        set_metrics(self)

        # build postprocess for infer
        if self.mode == 'infer':
            self.preprocess_func = create_operators(self.config["Infer"][
                "transforms"])
            self.postprocess_func = build_postprocess(self.config["Infer"][
                "PostProcess"])

    def train(self):
        assert self.mode == "train"
        print_batch_step = self.config.Global.print_batch_step
        save_interval = self.config.Global.save_interval
        best_metric = {
            "metric": 0.0,
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
        # global iter counter
        self.global_step = 0

        if self.config.Global.checkpoints is not None:
            metric_info = init_model(self.config.Global, self.model,
                                     self.optimizer)
            if metric_info is not None:
                best_metric.update(metric_info)

        self.max_iter = len(self.train_dataloader) - 1 if platform.system(
        ) == "Windows" else len(self.train_dataloader)
        for epoch_id in range(best_metric["epoch"] + 1,
                              self.config.Global.epochs + 1):

            # call train_epoch_func to train one epoch.
            self.train_epoch_func(self, epoch_id, print_batch_step)

            if self.use_dali:
                self.train_dataloader.reset()
            metric_msg = ", ".join([
                "{}: {:.5f}".format(key, self.output_info[key].avg)
                for key in self.output_info
            ])
            logger.info("[Train][Epoch {}/{}][Avg]{}".format(
                epoch_id, self.config.Global.epochs, metric_msg))
            self.output_info.clear()

            current_metric = 0.0
            # eval model and save model if possible
            if self.config.Global.eval_during_train and epoch_id % self.config.Global.eval_interval == 0:
                current_metric = self.eval(epoch_id)
                if current_metric > best_metric["metric"]:
                    best_metric["metric"] = current_metric
                    best_metric["epoch"] = epoch_id
                    save_load.save_model(
                        self.model,
                        self.optimizer,
                        best_metric,
                        self.output_dir,
                        model_name=self.config.Arch.name,
                        prefix="best_model")
                logger.info("[Eval][Epoch {}][best metric: {}]".format(
                    epoch_id, best_metric["metric"]))
                logger.scaler(
                    name="eval_acc",
                    value=current_metric,
                    step=epoch_id,
                    writer=self.vdl_writer)

            # save model every save_interval epoch
            if epoch_id % save_interval == 0:
                save_load.save_model(
                    self.model,
                    self.optimizer,
                    {"metric": current_metric,
                     "epoch": epoch_id},
                    self.output_dir,
                    model_name=self.config.Arch.name,
                    prefix="epoch_{}".format(epoch_id))

            # save the latest model
            save_load.save_model(
                self.model,
                self.optimizer, {"metric": current_metric,
                                 "epoch": epoch_id},
                self.output_dir,
                model_name=self.config.Arch.name,
                prefix="latest")

        if self.vdl_writer is not None:
            self.vdl_writer.close()

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_func(self, epoch_id)
        self.model.train()
        return eval_result

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
                out = self.model(batch_tensor)
                if isinstance(out, list):
                    out = out[0]
                if isinstance(out, dict) and "logits" in out:
                    out = out["logits"]
                if isinstance(out, dict) and "output" in out:
                    out = out["output"]
                result = self.postprocess_func(out, image_file_list)
                print(result)
                batch_data.clear()
                image_file_list.clear()

    def export(self):
        assert self.mode == "export"
        use_multilabel = self.config.Global.get("use_multilabel", False)
        model = ExportModel(self.config.Arch, self.model, use_multilabel)
        if self.config.Global.pretrained_model is not None:
            load_dygraph_pretrain(model.base_model,
                                  self.config.Global.pretrained_model)

        model.eval()
        save_path = os.path.join(self.config.Global.save_inference_dir,
                                 "inference")
        if model.quanter:
            model.quanter.save_quantized_model(
                model.base_model,
                save_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config.Global.image_shape,
                        dtype='float32')
                ])
        else:
            model = paddle.jit.to_static(
                model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config.Global.image_shape,
                        dtype='float32')
                ])
            paddle.jit.save(model, save_path)


class ExportModel(TheseusLayer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, models, use_multilabel):
        super().__init__()
        self.base_model = models
        # we should choose a final model to export
        if isinstance(self.base_model, DistillationModel):
            self.infer_model_name = config["infer_model_name"]
        else:
            self.infer_model_name = None

        self.infer_output_key = config.get("infer_output_key", None)
        if self.infer_output_key == "features" and isinstance(self.base_model,
                                                              RecModel):
            self.base_model.head = IdentityHead()
        if use_multilabel:
            self.out_act = nn.Sigmoid()
        else:
            if config.get("infer_add_softmax", True):
                self.out_act = nn.Softmax(axis=-1)
            else:
                self.out_act = None

    def eval(self):
        self.training = False
        for layer in self.sublayers():
            layer.training = False
            layer.eval()

    def forward(self, x):
        x = self.base_model(x)
        if isinstance(x, list):
            x = x[0]
        if self.infer_model_name is not None:
            x = x[self.infer_model_name]
        if self.infer_output_key is not None:
            x = x[self.infer_output_key]
        if self.out_act is not None:
            if isinstance(x, dict):
                x = x["logits"]
            x = self.out_act(x)
        return x
