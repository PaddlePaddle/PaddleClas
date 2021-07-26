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
        #  self.pact = self.config["Slim"].get("pact", False)
        self.pact = True
        if self.pact:
            quant_config["activation_preprocess_type"] = "PACT"
            self.quanter = QAT(config=quant_config)
            self.quanter.quantize(self.model)
            logger.info("QAT model summary:")
            paddle.summary(self.model, (1, 3, 224, 224))
        else:
            self.quanter = None

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

        assert self.quanter
        self.quanter.save_quantized_model(
            self.model,
            os.path.join(self.config["Global"]["save_inference_dir"],
                         "inference"),
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + config["Global"]["image_shape"],
                    dtype='float32')
            ])


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    trainer = Trainer_slim(config, mode="train")
    trainer.train()
