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
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random

from ppcls.utils import logger
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.arch import build_model, RecModel, DistillationModel, TheseusLayer
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

from ppcls.data.utils.get_image_list import get_image_list
from ppcls.data.postprocess import build_postprocess
from ppcls.data import create_operators
from .train import build_train_func
from .evaluation import build_eval_func
from ppcls.engine import evaluation
from ppcls.arch.gears.identity_head import IdentityHead


class Engine(object):
    def __init__(self, config, mode="train"):
        assert mode in ["train", "eval", "infer", "export"]
        self.mode = mode
        self.config = config

        # set seed
        self._init_seed()

        # init logger
        log_file = os.path.join(self.config['Global']['output_dir'],
                                self.config["Arch"]["name"], f"{mode}.log")
        init_logger(log_file=log_file)

        # set device
        self._init_device()

        # build model
        self.model = build_model(self.config, self.mode)

        # load_pretrain
        self._init_pretrained()

        self._init_amp()

        # init train_func and eval_func
        self.eval = build_eval_func(
            self.config, mode=self.mode, model=self.model)
        self.train = build_train_func(
            self.config, mode=self.mode, model=self.model, eval_func=self.eval)

        # for distributed
        self._init_dist()

        print_config(self.config)

    @paddle.no_grad()
    def infer(self):
        assert self.mode == "infer" and self.eval_mode == "classification"

        self.preprocess_func = create_operators(self.config["Infer"][
            "transforms"])
        self.postprocess_func = build_postprocess(self.config["Infer"][
            "PostProcess"])

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

    def export(self):
        assert self.mode == "export"
        use_multilabel = self.config["Global"].get(
            "use_multilabel",
            False) or "ATTRMetric" in self.config["Metric"]["Eval"][0]
        model = ExportModel(self.config["Arch"], self.model, use_multilabel)
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    model.base_model,
                    self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    model.base_model,
                    self.config["Global"]["pretrained_model"])

        model.eval()

        # for re-parameterization nets
        for layer in self.model.sublayers():
            if hasattr(layer, "re_parameterize") and not getattr(layer,
                                                                 "is_repped"):
                layer.re_parameterize()

        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")

        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + self.config["Global"]["image_shape"],
                    dtype='float32')
            ])
        if hasattr(model.base_model,
                   "quanter") and model.base_model.quanter is not None:
            model.base_model.quanter.save_quantized_model(model,
                                                          save_path + "_int8")
        else:
            paddle.jit.save(model, save_path)
        logger.info(
            f"Export succeeded! The inference model exported has been saved in \"{self.config['Global']['save_inference_dir']}\"."
        )

    def _init_seed(self):
        seed = self.config["Global"].get("seed", False)
        if dist.get_world_size() != 1:
            # if self.config["Global"]["distributed"]:
            # set different seed in different GPU manually in distributed environment
            if not seed:
                logger.warning(
                    "The random seed cannot be None in a distributed environment. Global.seed has been set to 42 by default"
                )
                self.config["Global"]["seed"] = seed = 42
            logger.info(
                f"Set random seed to ({int(seed)} + $PADDLE_TRAINER_ID) for different trainer"
            )
            dist_seed = int(seed) + dist.get_rank()
            paddle.seed(dist_seed)
            np.random.seed(dist_seed)
            random.seed(dist_seed)
        elif seed or seed == 0:
            assert isinstance(seed, int), "The 'seed' must be a integer!"
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

    def _init_device(self):
        device = self.config["Global"]["device"]
        assert device in ["cpu", "gpu", "xpu", "npu", "mlu", "ascend"]
        logger.info('train with paddle {} and device {}'.format(
            paddle.__version__, device))
        paddle.set_device(device)

    def _init_pretrained(self):
        if self.config["Global"]["pretrained_model"] is not None:
            if self.config["Global"]["pretrained_model"].startswith("http"):
                load_dygraph_pretrain_from_url(
                    [self.model, getattr(self, 'train_loss_func', None)],
                    self.config["Global"]["pretrained_model"])
            else:
                load_dygraph_pretrain(
                    [self.model, getattr(self, 'train_loss_func', None)],
                    self.config["Global"]["pretrained_model"])

    def _init_amp(self):
        if "AMP" in self.config and self.config["AMP"] is not None:
            paddle_version = paddle.__version__[:3]
            # paddle version < 2.3.0 and not develop
            if paddle_version not in ["2.3", "2.4", "0.0"]:
                msg = "When using AMP, PaddleClas release/2.6 and later version only support PaddlePaddle version >= 2.3.0."
                logger.error(msg)
                raise Exception(msg)

            AMP_RELATED_FLAGS_SETTING = {'FLAGS_max_inplace_grad_add': 8, }
            if paddle.is_compiled_with_cuda():
                AMP_RELATED_FLAGS_SETTING.update({
                    'FLAGS_cudnn_batchnorm_spatial_persistent': 1
                })
            paddle.set_flags(AMP_RELATED_FLAGS_SETTING)

            amp_level = self.config['AMP'].get("level", "O1").upper()
            if amp_level not in ["O1", "O2"]:
                msg = "[Parameter Error]: The optimize level of AMP only support 'O1' and 'O2'. The level has been set 'O1'."
                logger.warning(msg)
                self.config['AMP']["level"] = "O1"
                amp_level = "O1"

            amp_eval = self.config["AMP"].get("use_fp16_test", False)
            # TODO(gaotingquan): Paddle not yet support FP32 evaluation when training with AMPO2
            if self.mode == "train" and self.config["Global"].get(
                    "eval_during_train",
                    True) and amp_level == "O2" and amp_eval == False:
                msg = "PaddlePaddle only support FP16 evaluation when training with AMP O2 now. "
                logger.warning(msg)
                self.config["AMP"]["use_fp16_test"] = True
                amp_eval = True

            if self.mode == "train" or amp_eval:
                AMPForwardDecorator.amp_level = amp_level
                AMPForwardDecorator.amp_eval = amp_eval

    def _init_dist(self):
        # check the gpu num
        world_size = dist.get_world_size()
        self.config["Global"]["distributed"] = world_size != 1
        # TODO(gaotingquan):
        if self.mode == "train":
            std_gpu_num = 8 if isinstance(
                self.config["Optimizer"],
                dict) and self.config["Optimizer"]["name"] == "AdamW" else 4
            if world_size != std_gpu_num:
                msg = f"The training strategy provided by PaddleClas is based on {std_gpu_num} gpus. But the number of gpu is {world_size} in current training. Please modify the stategy (learning rate, batch size and so on) if use this config to train."
                logger.warning(msg)

        if self.config["Global"]["distributed"]:
            dist.init_parallel_env()
            self.model = paddle.DataParallel(self.model)
            if self.mode == 'train' and len(self.train_loss_func.parameters(
            )) > 0:
                self.train_loss_func = paddle.DataParallel(
                    self.train_loss_func)


class ExportModel(TheseusLayer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config, model, use_multilabel):
        super().__init__()
        self.base_model = model
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
