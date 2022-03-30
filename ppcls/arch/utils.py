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

import math
import types
from difflib import SequenceMatcher
from typing import Tuple

import paddle
import six
from paddle import Tensor

from . import backbone


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
    from ppcls.arch.backbone import (distilled_vision_transformer,
                                     vision_transformer)
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


# common weight init functions below
def _calculate_fan_in_and_fan_out(tensor: Tensor) -> Tuple[int, int]:
    """_calculate_fan_in_and_fan_out

    Args:
        tensor (Tensor): tensor.

    Returns:
        Tuple[int, int]: (fan_in, fan_out).
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions"
        )

    num_input_fmaps = tensor.shape[1]
    num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def kaiming_uniform_(tensor: Tensor,
                     a: int=0,
                     mode: str='fan_in',
                     nonlinearity: str='leaky_relu') -> Tensor:
    """inplace kaiming_uniform_ initialization

    Args:
        tensor (Tensor): tensor.
        a (int, optional): parameter a. Defaults to 0.
        mode (str, optional): mode. Defaults to 'fan_in'.
        nonlinearity (str, optional): nonlinearity. Defaults to 'leaky_relu'.

    Returns:
        Tensor: initialized tensor.
    """

    def _calculate_correct_fan(tensor: Tensor, mode: str) -> int:
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".
                             format(mode, valid_modes))

        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == 'fan_in' else fan_out

    def calculate_gain(nonlinearity: str, param=None) -> float:
        linear_fns = [
            'linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d',
            'conv_transpose2d', 'conv_transpose3d'
        ]
        if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
            return 1
        elif nonlinearity == 'tanh':
            return 5.0 / 3
        elif nonlinearity == 'relu':
            return math.sqrt(2.0)
        elif nonlinearity == 'leaky_relu':
            if param is None:
                negative_slope = 0.01
            elif not isinstance(param, bool) and isinstance(
                    param, int) or isinstance(param, float):
                negative_slope = param
            else:
                raise ValueError("negative_slope {} not a valid number".format(
                    param))
            return math.sqrt(2.0 / (1 + negative_slope**2))
        else:
            raise ValueError("Unsupported nonlinearity {}".format(
                nonlinearity))

    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(
        3.0) * std  # Calculate uniform bounds from standard deviation
    with paddle.no_grad():
        tensor.set_value(paddle.uniform(tensor.shape, min=-bound, max=bound))
        return tensor
