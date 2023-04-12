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

# reference: https://arxiv.org/abs/1611.05431

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from ....utils import logger
from ppcls.arch.init_weight import normal_init
from ..legendary_models import *
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {"moco_clas": "UNKNOWN"}

__all__ = list(MODEL_URLS.keys())


class ClasHead(nn.Layer):
    """Simple classifier head.
    """

    def __init__(self, with_avg_pool=False, in_channels=2048, class_num=1000):
        super(ClasHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = class_num

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(in_channels, class_num)
        # reset_parameters(self.fc_cls)
        normal_init(self.fc, mean=0.0, std=0.01, bias=0.0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = paddle.reshape(x, [-1, self.in_channels])
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
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


class Classification(nn.Layer):
    """
    Simple image classification.
    """

    def __init__(self, backbone, head, with_sobel=False):
        super(Classification, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, inputs):
        x = self.backbone(inputs)
        x = self.head(x)
        return x


def moco_clas(backbone, head, pretrained=False, use_ssld=False):
    backbone_config = backbone
    head_config = head
    backbone_name = backbone_config.pop('name')
    backbone = eval(backbone_name)(**backbone_config)
    # stop layer for backbone
    stop_layer_name = backbone_config.pop('stop_layer_name', None)
    if stop_layer_name:
        backbone.stop_after(stop_layer_name=stop_layer_name)
    # freeze specified layer before
    freeze_layer_name = backbone_config.pop('freeze_befor', None)
    if freeze_layer_name:
        ret = backbone.freeze_befor(freeze_layer_name)
        if ret:
            logger.info("moco_clas backbone successfully freeze param update befor the layer: " \
                        .format(freeze_layer_name))
        else:
            logger.error("moco_clas backbone failurely freeze param update befor the layer: " \
                        .format(freeze_layer_name))

    head_name = head_config.pop('name')
    head = eval(head_name)(**head_config)
    model = Classification(backbone=backbone, head=head)
    _load_pretrained(
        pretrained, model, MODEL_URLS["moco_clas"], use_ssld=use_ssld)
    return model
