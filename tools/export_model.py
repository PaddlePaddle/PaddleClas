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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import paddle
import paddle.nn as nn

from ppcls.utils import config
from ppcls.utils.logger import init_logger
from ppcls.utils.config import print_config
from ppcls.arch import build_model, RecModel, DistillationModel
from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.arch.gears.identity_head import IdentityHead


class ExportModel(nn.Layer):
    """
    ExportModel: add softmax onto the model
    """

    def __init__(self, config):
        super().__init__()
        self.base_model = build_model(config)

        # we should choose a final model to export
        if isinstance(self.base_model, DistillationModel):
            self.infer_model_name = config["infer_model_name"]
        else:
            self.infer_model_name = None

        self.infer_output_key = config.get("infer_output_key", None)
        if self.infer_output_key == "features" and isinstance(self.base_model,
                                                              RecModel):
            self.base_model.head = IdentityHead()
        if config.get("infer_add_softmax", True):
            self.softmax = nn.Softmax(axis=-1)
        else:
            self.softmax = None

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
        if self.softmax is not None:
            x = self.softmax(x)
        return x


if __name__ == "__main__":
    args = config.parse_args()
    config = config.get_config(
        args.config, overrides=args.override, show=False)
    log_file = os.path.join(config['Global']['output_dir'],
                            config["Arch"]["name"], "export.log")
    init_logger(name='root', log_file=log_file)
    print_config(config)

    # set device
    assert config["Global"]["device"] in ["cpu", "gpu", "xpu"]
    device = paddle.set_device(config["Global"]["device"])
    model = ExportModel(config["Arch"])
    if config["Global"]["pretrained_model"] is not None:
        load_dygraph_pretrain(model.base_model,
                              config["Global"]["pretrained_model"])

    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + config["Global"]["image_shape"],
                dtype='float32')
        ])
    paddle.jit.save(model,
                    os.path.join(config["Global"]["save_inference_dir"],
                                 "inference"))
