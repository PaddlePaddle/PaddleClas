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
from ppcls.utils.initializer import normal_
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
        normal_(self.fc, mean=0.0, std=0.01, bias=0.0)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = paddle.reshape(x, [-1, self.in_channels])
        x = self.fc(x)
        return x


def _load_pretrained(pretrained_config, model, use_ssld=False):
    if pretrained_config is not None:
        if pretrained_config.startswith("http"):
            load_dygraph_pretrain_from_url(model.base_model, pretrained_config)
        else:
            load_dygraph_pretrain(model.base_model, pretrained_config)


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


def freeze_batchnorm_statictis(layer):
    def freeze_bn(layer):
        if isinstance(layer, nn.BatchNorm):
            layer._use_global_stats = True


def freeze_params(model):
    from ppcls.arch.backbone.legendary_models.resnet import ConvBNLayer, BottleneckBlock
    for item in ['stem', 'max_pool', 'blocks', 'avg_pool']:
        m = getattr(model, item)
        if isinstance(m, nn.Sequential):
            for item in m:
                if isinstance(item, ConvBNLayer):
                    print(item.bn)
                    freeze_batchnorm_statictis(item.bn)

                if isinstance(item, BottleneckBlock):
                    freeze_batchnorm_statictis(item.conv0.bn)
                    freeze_batchnorm_statictis(item.conv1.bn)
                    freeze_batchnorm_statictis(item.conv2.bn)
                    if hasattr(item, 'short'):
                        freeze_batchnorm_statictis(item.short.bn)

        for param in m.parameters():
            param.trainable = False


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
            logger.info(
                "moco_clas backbone successfully freeze param update befor the layer: {}".
                format(freeze_layer_name))
        else:
            logger.error(
                "moco_clas backbone failurely freeze param update befor the layer: {}".
                format(freeze_layer_name))

    freeze_params(backbone)
    head_name = head_config.pop('name')
    head = eval(head_name)(**head_config)
    model = Classification(backbone=backbone, head=head)

    # load pretrain_moco_model weight
    pretrained_config = backbone_config.pop('pretrained_model')
    _load_pretrained(pretrained_config, model, use_ssld=use_ssld)
    return model
