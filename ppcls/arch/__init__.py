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

import copy
import importlib

import paddle.nn as nn

from . import backbone
from . import head

from .backbone import *
from .head import *
from .utils import *

__all__ = ["build_model", "RecModel"]


def build_model(config):
    config = copy.deepcopy(config)
    model_type = config.pop("name")
    mod = importlib.import_module(__name__)
    arch = getattr(mod, model_type)(**config)
    return arch


class RecModel(nn.Layer):
    def __init__(self, **config):
        super().__init__()

        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = eval(backbone_name)(**backbone_config)

        assert "Stoplayer" in config, "Stoplayer should be specified in retrieval task \
                please specified a Stoplayer config"

        stop_layer_config = config["Stoplayer"]
        self.backbone.stop_after(stop_layer_config["name"])

        if stop_layer_config.get("embedding_size", 0) > 0:
            self.neck = nn.Linear(stop_layer_config["output_dim"],
                                  stop_layer_config["embedding_size"])
            embedding_size = stop_layer_config["embedding_size"]
        else:
            self.neck = None
            embedding_size = stop_layer_config["output_dim"]

        assert "Head" in config, "Head should be specified in retrieval task \
                please specify a Head config"

        config["Head"]["embedding_size"] = embedding_size
        self.head = build_head(config["Head"])

    def forward(self, x, label):
        x = self.backbone(x)
        if self.neck is not None:
            x = self.neck(x)
        y = self.head(x, label)
        return {"features": x, "logits": y}
