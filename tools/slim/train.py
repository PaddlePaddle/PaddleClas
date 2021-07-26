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

import os
import sys

import paddle
import paddleslim
from paddle.jit import to_static
from paddleslim.analysis import dygraph_flops as flops

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
from paddleslim.dygraph.quant import QAT

from ppcls.engine.trainer import Trainer
from ppcls.utils import config, logger
from ppcls.utils.save_load import load_dygraph_pretrain

quant_config = {
    # weight preprocess type, default is None and no preprocessing is performed. 
    'weight_preprocess_type': None,
    # activation preprocess type, default is None and no preprocessing is performed.
    'activation_preprocess_type': None,
    # weight quantize type, default is 'channel_wise_abs_max'
    'weight_quantize_type': 'channel_wise_abs_max',
    # activation quantize type, default is 'moving_average_abs_max'
    'activation_quantize_type': 'moving_average_abs_max',
    # weight quantize bit num, default is 8
    'weight_bits': 8,
    # activation quantize bit num, default is 8
    'activation_bits': 8,
    # data type after quantization, such as 'uint8', 'int8', etc. default is 'int8'
    'dtype': 'int8',
    # window size for 'range_abs_max' quantization. default is 10000
    'window_size': 10000,
    # The decay coefficient of moving average, default is 0.9
    'moving_rate': 0.9,
    # for dygraph quantization, layers of type in quantizable_layer_type will be quantized
    'quantizable_layer_type': ['Conv2D', 'Linear'],
}


class Trainer_slim(Trainer):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode)
        pact = self.config["Slim"].get("quant", False)
        self.pact = pact.get("name", False) if pact else pact

        if self.pact and str(self.pact.lower()) != 'pact':
            raise RuntimeError("The quantization only support 'PACT'!")
        if pact:
            quant_config["activation_preprocess_type"] = "PACT"
            self.quanter = QAT(config=quant_config)
            self.quanter.quantize(self.model)
            logger.info("QAT model summary:")
            paddle.summary(self.model, (1, 3, 224, 224))
        else:
            self.quanter = None

        prune_config = self.config["Slim"].get("prune", False)
        if prune_config:
            if prune_config["name"].lower() not in ["fpgm", "l1_norm"]:
                raise RuntimeError(
                    "The prune methods only support 'fpgm' and 'l1_norm'")
            else:
                logger.info("FLOPs before pruning: {}GFLOPs".format(
                    flops(self.model, [1] + self.config["Global"][
                        "image_shape"]) / 1000000))
                self.model.eval()

                if prune_config["name"].lower() == "fpgm":
                    self.model.eval()
                    self.pruner = paddleslim.dygraph.FPGMFilterPruner(
                        self.model, [1] + self.config["Global"]["image_shape"])
                else:
                    self.pruner = paddleslim.dygraph.L1NormFilterPruner(
                        self.model, [1] + self.config["Global"]["image_shape"])
                self.prune_model()
        else:
            self.pruner = None

        if self.quanter is None and self.pruner is None:
            logger.info("Training without slim")

    def train(self):
        super().train()
        if self.config["Global"].get("save_inference_dir", None):
            self.export_inference_model()

    def export_inference_model(self):
        if os.path.exists(
                os.path.join(self.output_dir, self.config["Arch"]["name"],
                             "best_model.pdparams")):
            load_dygraph_pretrain(self.model,
                                  os.path.join(self.output_dir,
                                               self.config["Arch"]["name"],
                                               "best_model"))
        elif self.config["Global"].get(
                "pretraine_model", False) and os.path.exists(self.config[
                    "Global"]["pretraine_model"] + ".pdparams"):
            load_dygraph_pretrain(self.model,
                                  self.config["Global"]["pretraine_model"])
        else:
            raise RuntimeError(
                "The best_model or pretraine_model should exist to generate inference model"
            )
        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")
        if self.quanter:
            self.quanter.save_quantized_model(
                self.model,
                save_path,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + config["Global"]["image_shape"],
                        dtype='float32')
                ])
        else:
            model = to_static(
                self.model,
                input_spec=[
                    paddle.static.InputSpec(
                        shape=[None] + self.config["Global"]["image_shape"],
                        dtype='float32',
                        name="image")
                ])
            paddle.jit.save(model, save_path)

    def prune_model(self):
        params = []
        for sublayer in self.model.sublayers():
            for param in sublayer.parameters(include_sublayers=False):
                if isinstance(sublayer, paddle.nn.Conv2D):
                    params.append(param.name)
        ratios = {}
        for param in params:
            ratios[param] = self.config["Slim"]["prune"]["pruned_ratio"]
        plan = self.pruner.prune_vars(ratios, [0])

        logger.info("FLOPs after pruning: {}GFLOPs; pruned ratio: {}".format(
            flops(self.model, [1] + self.config["Global"]["image_shape"]) /
            1000000, plan.pruned_flops))

        for param in self.model.parameters():
            if "conv2d" in param.name:
                logger.info("{}\t{}".format(param.name, param.shape))

        self.model.train()


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    trainer = Trainer_slim(config, mode="train")
    trainer.train()
