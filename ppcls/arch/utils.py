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

import six
import types
import paddle
from difflib import SequenceMatcher

from . import backbone
from typing import Any, Dict, Union


def get_architectures():
    """
    get all of model architectures
    """
    names = []
    for k, v in backbone.__dict__.items():
        if isinstance(v, (types.FunctionType, six.class_types)):
            names.append(k)
    return names


def get_blacklist_model_in_static_mode():
    from ppcls.arch.backbone import distilled_vision_transformer
    from ppcls.arch.backbone import vision_transformer
    blacklist = distilled_vision_transformer.__all__ + vision_transformer.__all__
    return blacklist


def similar_architectures(name='', names=[], thresh=0.1, topk=10):
    """
    inferred similar architectures
    """
    scores = []
    for idx, n in enumerate(names):
        if n.startswith('__'):
            continue
        score = SequenceMatcher(None, n.lower(), name.lower()).quick_ratio()
        if score > thresh:
            scores.append((idx, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    similar_names = [names[s[0]] for s in scores[:min(topk, len(scores))]]
    return similar_names


def get_param_attr_dict(ParamAttr_config: Union[None, bool, Dict[str, Dict]]
                        ) -> Union[None, bool, paddle.ParamAttr]:
    """parse ParamAttr from an dict

    Args:
        ParamAttr_config (Union[None, bool, Dict[str, Dict]]): ParamAttr configure

    Returns:
        Union[None, bool, paddle.ParamAttr]: Generated ParamAttr
    """
    if ParamAttr_config is None:
        return None
    if isinstance(ParamAttr_config, bool):
        return ParamAttr_config
    ParamAttr_dict = {}
    if 'initializer' in ParamAttr_config:
        initializer_cfg = ParamAttr_config.get('initializer')
        if 'name' in initializer_cfg:
            initializer_name = initializer_cfg.pop('name')
            ParamAttr_dict['initializer'] = getattr(
                paddle.nn.initializer, initializer_name)(**initializer_cfg)
        else:
            raise ValueError(f"'name' must specified in initializer_cfg")
    if 'learning_rate' in ParamAttr_config:
        # NOTE: only support an single value now
        learning_rate_value = ParamAttr_config.get('learning_rate')
        if isinstance(learning_rate_value, (int, float)):
            ParamAttr_dict['learning_rate'] = learning_rate_value
        else:
            raise ValueError(
                f"learning_rate_value must be float or int, but got {type(learning_rate_value)}"
            )
    if 'regularizer' in ParamAttr_config:
        regularizer_cfg = ParamAttr_config.get('regularizer')
        if 'name' in regularizer_cfg:
            # L1Decay or L2Decay
            regularizer_name = regularizer_cfg.pop('name')
            ParamAttr_dict['regularizer'] = getattr(
                paddle.regularizer, regularizer_name)(**regularizer_cfg)
        else:
            raise ValueError(f"'name' must specified in regularizer_cfg")
    return paddle.ParamAttr(**ParamAttr_dict)
