from __future__ import absolute_import, division, print_function

import math

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Uniform
from ppcls.utils.save_load import (load_dygraph_pretrain,
                                   load_dygraph_pretrain_from_url)

__all__ = ["LeNetPlus"]


def _calculate_fan_in_and_fan_out(tensor):
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


def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    def _calculate_correct_fan(tensor: paddle.Tensor, mode: str):
        mode = mode.lower()
        valid_modes = ['fan_in', 'fan_out']
        if mode not in valid_modes:
            raise ValueError("Mode {} not supported, please use one of {}".
                             format(mode, valid_modes))

        fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
        return fan_in if mode == 'fan_in' else fan_out

    def calculate_gain(nonlinearity: str, param=None):
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


class LeNet(nn.Layer):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, num_classes):
        super(LeNet, self).__init__()
        self.conv1_1 = nn.Conv2D(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2D(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()

        self.conv2_1 = nn.Conv2D(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2D(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()

        self.conv3_1 = nn.Conv2D(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2D(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()

        self.fc1 = nn.Linear(128 * 3 * 3, 2)

        self.prelu_fc1 = nn.PReLU()
        self.fc2 = nn.Linear(2, num_classes)

        def _init_func(m):
            if isinstance(m, nn.Conv2D):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    uniform_ = Uniform(-bound, bound)
                    uniform_(m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    uniform_ = Uniform(-bound, bound)
                    uniform_(m.bias)

        self.apply(_init_func)

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)

        x = x.reshape([-1, 128 * 3 * 3])
        x = self.prelu_fc1(self.fc1(x))  # neck, backbone->features
        y = self.fc2(x)  # head features->logits

        return {'features': x, 'logits': y}


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


def LeNetPlus(pretrained=False, use_ssld=False, **kwargs):
    model = LeNet(**kwargs)
    _load_pretrained(pretrained, model, None, use_ssld=use_ssld)
    return model
