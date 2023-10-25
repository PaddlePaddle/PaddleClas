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

from .arcmargin import ArcMargin
from .cosmargin import CosMargin
from .circlemargin import CircleMargin
from .fc import FC
from .vehicle_neck import VehicleNeck
from paddle.nn import Tanh, Identity
from .bnneck import BNNeck
from .adamargin import AdaMargin
from .frfn_neck import FRFNNeck
from .metabnneck import MetaBNNeck
from .ml_decoder import MLDecoder

__all__ = ['build_gear', 'add_ml_decoder_head']


def build_gear(config):
    support_dict = [
        'ArcMargin', 'CosMargin', 'CircleMargin', 'FC', 'VehicleNeck', 'Tanh',
        'BNNeck', 'AdaMargin', 'FRFNNeck', 'MetaBNNeck'
    ]
    module_name = config.pop('name')
    assert module_name in support_dict, Exception(
        'head only support {}'.format(support_dict))
    module_class = eval(module_name)(**config)
    return module_class


def add_ml_decoder_head(model, config):
    if 'class_num' not in config:
        if hasattr(model, 'class_num'):
            config['class_num'] = model.class_num
        else:
            raise AttributeError(
                'Please manually add parameter `class_num` '
                'for MLDecoder in the config file.')

    # remove_layers: list of layer names that need to be deleted from backbone
    if 'remove_layers' in config:
        remove_layers = config.pop('remove_layers')
    else:
        remove_layers = ['avg_pool', 'flatten']
    for remove_layer in remove_layers:
        if hasattr(model, remove_layer):
            delattr(model, remove_layer)
            setattr(model, remove_layer, Identity())
        else:
            raise AttributeError(
                f"{remove_layer} does not have attribute the model.")

    # replace_layer: layer name that need to be replaced in backbone
    if 'replace_layer' in config:
        replace_layer = config.pop('replace_layer')
    else:
        replace_layer = 'fc'
    if hasattr(model, replace_layer):
        delattr(model, replace_layer)
        setattr(model, replace_layer, MLDecoder(**config))
    else:
        raise AttributeError(
            f"{replace_layer} does not have attribute the model.")
