from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.nn as nn
from paddle import Tensor
from ppcls.arch.utils import _calculate_fan_in_and_fan_out, kaiming_uniform_
from ppcls.utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

__all__ = ["LeNetPlus"]


class LeNet(nn.Layer):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self):
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

        self.max_pool2d = nn.MaxPool2D(kernel_size=2)

        def _init_func(m: nn.Layer):
            if isinstance(m, nn.Conv2D):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.initializer.Uniform(-bound, bound)(m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = _calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.initializer.Uniform(-bound, bound)(m.bias)

        self.apply(_init_func)

    def forward(self, x: Tensor) -> Tensor:
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = self.max_pool2d(x, 2)

        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = self.max_pool2d(x, 2)

        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = self.max_pool2d(x, 2)

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


def LeNetPlus(pretrained=False, use_ssld=False, **kwargs):
    model = LeNet(**kwargs)
    _load_pretrained(pretrained, model, None, use_ssld=use_ssld)
    return model
