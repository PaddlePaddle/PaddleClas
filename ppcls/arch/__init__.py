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
from . import backbone

from .backbone import *
from ppcls.arch.loss_metrics.loss import *
from .utils import *


def build_arch(config):
    config = copy.deepcopy(config)
    arch_type = config.pop("type")
    module = sys.modules[__name__]
    arch = getattr(module)(config)
    return arch


class BaseModel(nn.Layer):
    def __init__(self, config):
        super().__init__()
        backbone_config = config["Backbone"]
        backbone_type = backbone_config.pop("type")
        self.backbone = getattr(backbone)(**backbone_config)

        if "Neck" in config:
            neck_config = config["Neck"]
            neck_type = neck_config.pop("type")
            self.neck = getattr(backbone)(**neck_config)
        else:
            self.neck = None

        if "Head" in config:
            head_config = config["Head"]
            head_type = head_config.pop("type")
            self.head = getattr(head)(**head_config)
        else:
            self.head = None

    def forward(self, x):
        y = self.backbone(x)
        if self.neck is not None:
            y = self.neck(y)
        if self.head is not None:
            y = self.head(y)
        return y
