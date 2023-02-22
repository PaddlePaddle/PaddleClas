#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
from paddle.jit import to_static
from paddle.static import InputSpec

from . import backbone as backbone_zoo
from .gears import build_gear
from .utils import *
from .backbone.base.theseus_layer import TheseusLayer
from ..utils import logger
from ..utils.save_load import load_dygraph_pretrain
from .slim import prune_model, quantize_model
from .distill.afd_attention import LinearTransformStudent, LinearTransformTeacher

__all__ = ["build_model", "RecModel", "DistillationModel", "AttentionModel"]


def build_model(config, mode="train"):
    arch_config = copy.deepcopy(config["Arch"])
    model_type = arch_config.pop("name")
    use_sync_bn = arch_config.pop("use_sync_bn", False)

    if hasattr(backbone_zoo, model_type):
        model = ClassModel(model_type, **arch_config)
    else:
        model = getattr(sys.modules[__name__], model_type)("ClassModel",
                                                           **arch_config)

    if use_sync_bn:
        if config["Global"]["device"] == "gpu":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        else:
            msg = "SyncBatchNorm can only be used on GPU device. The releated setting has been ignored."
            logger.warning(msg)

    if isinstance(model, TheseusLayer):
        prune_model(config, model)
        quantize_model(config, model, mode)

    # set @to_static for benchmark, skip this by default.
    model = apply_to_static(config, model)

    return model


def apply_to_static(config, model):
    support_to_static = config['Global'].get('to_static', False)

    if support_to_static:
        specs = None
        if 'image_shape' in config['Global']:
            specs = [InputSpec([None] + config['Global']['image_shape'])]
            specs[0].stop_gradient = True
        model = to_static(model, input_spec=specs)
        logger.info("Successfully to apply @to_static with specs: {}".format(
            specs))
    return model


# TODO(gaotingquan): export model
class ClassModel(TheseusLayer):
    def __init__(self, model_type, **config):
        super().__init__()
        if model_type == "ClassModel":
            backbone_config = config["Backbone"]
            backbone_name = backbone_config.pop("name")
        else:
            backbone_name = model_type
            backbone_config = config
        self.backbone = getattr(backbone_zoo, backbone_name)(**backbone_config)

    def forward(self, batch):
        x, label = batch[0], batch[1]
        return self.backbone(x)


class RecModel(TheseusLayer):
    def __init__(self, **config):
        super().__init__()
        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = getattr(backbone_zoo, backbone_name)(**backbone_config)
        self.head_feature_from = config.get('head_feature_from', 'neck')

        if "BackboneStopLayer" in config:
            backbone_stop_layer = config["BackboneStopLayer"]["name"]
            self.backbone.stop_after(backbone_stop_layer)

        if "Neck" in config:
            self.neck = build_gear(config["Neck"])
        else:
            self.neck = None

        if "Head" in config:
            self.head = build_gear(config["Head"])
        else:
            self.head = None

    def forward(self, batch):
        x, label = batch[0], batch[1]
        out = dict()
        x = self.backbone(x)
        out["backbone"] = x
        if self.neck is not None:
            feat = self.neck(x)
            out["neck"] = feat
        out["features"] = out['neck'] if self.neck else x
        if self.head is not None:
            if self.head_feature_from == 'backbone':
                y = self.head(out['backbone'], label)
            elif self.head_feature_from == 'neck':
                y = self.head(out['features'], label)
            out["logits"] = y
        return out


class DistillationModel(nn.Layer):
    def __init__(self,
                 models=None,
                 pretrained_list=None,
                 freeze_params_list=None,
                 **kargs):
        super().__init__()
        assert isinstance(models, list)
        self.model_list = []
        self.model_name_list = []
        if pretrained_list is not None:
            assert len(pretrained_list) == len(models)

        if freeze_params_list is None:
            freeze_params_list = [False] * len(models)
        assert len(freeze_params_list) == len(models)
        for idx, model_config in enumerate(models):
            assert len(model_config) == 1
            key = list(model_config.keys())[0]
            model_config = model_config[key]
            model_name = model_config.pop("name")
            model = eval(model_name)(**model_config)

            if freeze_params_list[idx]:
                for param in model.parameters():
                    param.trainable = False
            self.model_list.append(self.add_sublayer(key, model))
            self.model_name_list.append(key)

        if pretrained_list is not None:
            for idx, pretrained in enumerate(pretrained_list):
                if pretrained is not None:
                    load_dygraph_pretrain(
                        self.model_name_list[idx], path=pretrained)

    def forward(self, batch):
        x, label = batch[0], batch[1]
        result_dict = dict()
        for idx, model_name in enumerate(self.model_name_list):
            if label is None:
                result_dict[model_name] = self.model_list[idx](x)
            else:
                result_dict[model_name] = self.model_list[idx](x, label)
        return result_dict


class AttentionModel(DistillationModel):
    def __init__(self,
                 models=None,
                 pretrained_list=None,
                 freeze_params_list=None,
                 **kargs):
        super().__init__(models, pretrained_list, freeze_params_list, **kargs)

    def forward(self, batch):
        x, label = batch[0], batch[1]
        result_dict = dict()
        out = x
        for idx, model_name in enumerate(self.model_name_list):
            if label is None:
                out = self.model_list[idx](out)
                result_dict.update(out)
            else:
                out = self.model_list[idx](out, label)
                result_dict.update(out)
        return result_dict
