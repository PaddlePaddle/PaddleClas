#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import sys
import copy

import paddle.nn as nn

from . import backbone

from .backbone import *
from ppcls.arch.loss_metrics.loss import *
from .utils import *


def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = sys.modules[__name__]
    arch = getattr(mod, model_type)(**config)
    return arch


class RecModel(nn.Layer):
    def __init__(self, **config):
        super().__init__()
        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = getattr(backbone_name)(**backbone_config)

        if "Neck" in config:
            neck_config = config["Neck"]
            neck_name = neck_config.pop("name")
            self.neck = getattr(neck_name)(**neck_config)
        else:
            self.neck = None

        if "Head" in config:
            head_config = config["Head"]
            head_name = head_config.pop("name")
            self.head = getattr(head_name)(**head_config)
        else:
            self.head = None

    def forward(self, x):
        y = self.backbone(x)
        if self.neck is not None:
            y = self.neck(y)
        if self.head is not None:
            y = self.head(y)
        return y
