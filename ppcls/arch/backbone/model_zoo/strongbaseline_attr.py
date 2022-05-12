# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Conv2D, BatchNorm, Linear, Dropout
from paddle.nn import AdaptiveAvgPool2D, MaxPool2D, AvgPool2D
from paddle.nn.initializer import Uniform

import math

from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url, get_weights_path_from_url
from ..legendary_models.resnet import ResNet50

MODEL_URLS = {"StrongBaselineAttr": "strongbaseline_attr_clas", }

__all__ = list(MODEL_URLS.keys())


class StrongBaselinePAR(nn.Layer):
    def __init__(
            self,
            **config, ):
        """
        A strong baseline for Pedestrian Attribute Recognition, see https://arxiv.org/abs/2107.03576 

        Args:
            backbone (object): backbone instance
            classifier (object): classifier instance
            loss (object): loss instance
        """
        super(StrongBaselinePAR, self).__init__()
        backbone_config = config["Backbone"]
        backbone_name = backbone_config.pop("name")
        self.backbone = eval(backbone_name)(**backbone_config)

    def forward(self, x):
        fc_feat = self.backbone(x)
        output = F.sigmoid(fc_feat)
        return fc_feat


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def load_pretrained(model, local_weight_path):
    # local_weight_path = get_weights_path_from_url(model_url).replace(
    #     ".pdparams", "")
    param_state_dict = paddle.load(local_weight_path + ".pdparams")
    model_dict = model.state_dict()
    model_dict_keys = list(model_dict.keys())
    param_state_dict_keys = list(param_state_dict.keys())

    # assert(len(model_dict_keys) == len(param_state_dict_keys)), "{} == {}".format(len(model_dict_keys), len(param_state_dict_keys))
    for idx in range(len(model_dict.keys())):
        model_key = model_dict_keys[idx]
        param_key = param_state_dict_keys[idx]
        if model_dict[model_key].shape == param_state_dict[param_key].shape:
            model_dict[model_key] = param_state_dict[param_key]
        else:
            print("miss match idx: {} weights: {} vs {}; {} vs {}".format(
                idx, model_key, param_key, model_dict[
                    model_key].shape, param_state_dict[param_key].shape))
    model.set_dict(model_dict)


def StrongBaselineAttr(pretrained=True, use_ssld=False, **kwargs):
    model = StrongBaselinePAR(**kwargs)
    _load_pretrained(MODEL_URLS["StrongBaselineAttr"], model, None, None)
    # load_pretrained(model, MODEL_URLS["StrongBaselineAttr"])
    return model
