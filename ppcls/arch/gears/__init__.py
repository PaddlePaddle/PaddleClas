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
    for layer_name, new_layer in zip(
            ["avg_pool", "flatten", "fc"],
            [Identity(), Identity(), MLDecoder(**config)]):

        if hasattr(model, layer_name):
            delattr(model, layer_name)
            setattr(model, layer_name, new_layer)
        else:
            raise AttributeError(
                "Please carefully check that the last three layers of the model "
                "you need to add a are `avg_pool`, `flatten`, and `fc`.")
