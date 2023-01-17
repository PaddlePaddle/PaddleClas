# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/2103.13425, https://github.com/DingXiaoH/DiverseBranchBlock

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
from .dbb_transforms import *


def conv_bn(in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            padding_mode='zeros'):
    conv_layer = nn.Conv2D(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias_attr=False,
        padding_mode=padding_mode)
    bn_layer = nn.BatchNorm2D(num_features=out_channels)
    se = nn.Sequential()
    se.add_sublayer('conv', conv_layer)
    se.add_sublayer('bn', bn_layer)
    return se


class IdentityBasedConv1x1(nn.Conv2D):
    def __init__(self, channels, groups=1):
        super(IdentityBasedConv1x1, self).__init__(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias_attr=False)

        assert channels % groups == 0
        input_dim = channels // groups
        id_value = np.zeros((channels, input_dim, 1, 1))
        for i in range(channels):
            id_value[i, i % input_dim, 0, 0] = 1
        self.id_tensor = paddle.to_tensor(id_value)
        self.weight.set_value(paddle.zeros_like(self.weight))

    def forward(self, input):
        kernel = self.weight + self.id_tensor
        result = F.conv2d(
            input,
            kernel,
            None,
            stride=1,
            padding=0,
            dilation=self._dilation,
            groups=self._groups)
        return result

    def get_actual_kernel(self):
        return self.weight + self.id_tensor


class BNAndPad(nn.Layer):
    def __init__(self,
                 pad_pixels,
                 num_features,
                 epsilon=1e-5,
                 momentum=0.1,
                 last_conv_bias=None,
                 bn=nn.BatchNorm2D):
        super().__init__()
        self.bn = bn(num_features, momentum=momentum, epsilon=epsilon)
        self.pad_pixels = pad_pixels
        self.last_conv_bias = last_conv_bias

    def forward(self, input):
        output = self.bn(input)
        if self.pad_pixels > 0:
            bias = -self.bn._mean
            if self.last_conv_bias is not None:
                bias += self.last_conv_bias
            pad_values = self.bn.bias + self.bn.weight * (
                bias / paddle.sqrt(self.bn._variance + self.bn._epsilon))
            ''' pad '''
            # TODO: n,h,w,c format is not supported yet
            n, c, h, w = output.shape
            values = pad_values.reshape([1, -1, 1, 1])
            w_values = values.expand([n, -1, self.pad_pixels, w])
            x = paddle.concat([w_values, output, w_values], axis=2)
            h = h + self.pad_pixels * 2
            h_values = values.expand([n, -1, h, self.pad_pixels])
            x = paddle.concat([h_values, x, h_values], axis=3)
            output = x
        return output

    @property
    def weight(self):
        return self.bn.weight

    @property
    def bias(self):
        return self.bn.bias

    @property
    def _mean(self):
        return self.bn._mean

    @property
    def _variance(self):
        return self.bn._variance

    @property
    def _epsilon(self):
        return self.bn._epsilon


class DiverseBranchBlock(nn.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 is_repped=False,
                 single_init=False,
                 **kwargs):
        super().__init__()

        padding = (filter_size - 1) // 2
        dilation = 1

        in_channels = num_channels
        out_channels = num_filters
        kernel_size = filter_size
        internal_channels_1x1_3x3 = None
        nonlinear = act

        self.is_repped = is_repped

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nn.ReLU()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if is_repped:
            self.dbb_reparam = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=True)
        else:
            self.dbb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups)

            self.dbb_avg = nn.Sequential()
            if groups < out_channels:
                self.dbb_avg.add_sublayer(
                    'conv',
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias_attr=False))
                self.dbb_avg.add_sublayer(
                    'bn',
                    BNAndPad(
                        pad_pixels=padding, num_features=out_channels))
                self.dbb_avg.add_sublayer(
                    'avg',
                    nn.AvgPool2D(
                        kernel_size=kernel_size, stride=stride, padding=0))
                self.dbb_1x1 = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    groups=groups)
            else:
                self.dbb_avg.add_sublayer(
                    'avg',
                    nn.AvgPool2D(
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding))

            self.dbb_avg.add_sublayer('avgbn', nn.BatchNorm2D(out_channels))

            if internal_channels_1x1_3x3 is None:
                internal_channels_1x1_3x3 = in_channels if groups < out_channels else 2 * in_channels  # For mobilenet, it is better to have 2X internal channels

            self.dbb_1x1_kxk = nn.Sequential()
            if internal_channels_1x1_3x3 == in_channels:
                self.dbb_1x1_kxk.add_sublayer(
                    'idconv1',
                    IdentityBasedConv1x1(
                        channels=in_channels, groups=groups))
            else:
                self.dbb_1x1_kxk.add_sublayer(
                    'conv1',
                    nn.Conv2D(
                        in_channels=in_channels,
                        out_channels=internal_channels_1x1_3x3,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        groups=groups,
                        bias_attr=False))
            self.dbb_1x1_kxk.add_sublayer(
                'bn1',
                BNAndPad(
                    pad_pixels=padding,
                    num_features=internal_channels_1x1_3x3))
            self.dbb_1x1_kxk.add_sublayer(
                'conv2',
                nn.Conv2D(
                    in_channels=internal_channels_1x1_3x3,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias_attr=False))
            self.dbb_1x1_kxk.add_sublayer('bn2', nn.BatchNorm2D(out_channels))

        #   The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #   Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def forward(self, inputs):
        if self.is_repped:
            return self.nonlinear(self.dbb_reparam(inputs))

        out = self.dbb_origin(inputs)
        if hasattr(self, 'dbb_1x1'):
            out += self.dbb_1x1(inputs)
        out += self.dbb_avg(inputs)
        out += self.dbb_1x1_kxk(inputs)
        return self.nonlinear(out)

    def init_gamma(self, gamma_value):
        if hasattr(self, "dbb_origin"):
            paddle.nn.init.constant_(self.dbb_origin.bn.weight, gamma_value)
        if hasattr(self, "dbb_1x1"):
            paddle.nn.init.constant_(self.dbb_1x1.bn.weight, gamma_value)
        if hasattr(self, "dbb_avg"):
            paddle.nn.init.constant_(self.dbb_avg.avgbn.weight, gamma_value)
        if hasattr(self, "dbb_1x1_kxk"):
            paddle.nn.init.constant_(self.dbb_1x1_kxk.bn2.weight, gamma_value)

    def single_init(self):
        self.init_gamma(0.0)
        if hasattr(self, "dbb_origin"):
            paddle.nn.init.constant_(self.dbb_origin.bn.weight, 1.0)

    def get_equivalent_kernel_bias(self):
        k_origin, b_origin = transI_fusebn(self.dbb_origin.conv.weight,
                                           self.dbb_origin.bn)

        if hasattr(self, 'dbb_1x1'):
            k_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight,
                                         self.dbb_1x1.bn)
            k_1x1 = transVI_multiscale(k_1x1, self.kernel_size)
        else:
            k_1x1, b_1x1 = 0, 0

        if hasattr(self.dbb_1x1_kxk, 'idconv1'):
            k_1x1_kxk_first = self.dbb_1x1_kxk.idconv1.get_actual_kernel()
        else:
            k_1x1_kxk_first = self.dbb_1x1_kxk.conv1.weight
        k_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(k_1x1_kxk_first,
                                                         self.dbb_1x1_kxk.bn1)
        k_1x1_kxk_second, b_1x1_kxk_second = transI_fusebn(
            self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2)
        k_1x1_kxk_merged, b_1x1_kxk_merged = transIII_1x1_kxk(
            k_1x1_kxk_first,
            b_1x1_kxk_first,
            k_1x1_kxk_second,
            b_1x1_kxk_second,
            groups=self.groups)

        k_avg = transV_avg(self.out_channels, self.kernel_size, self.groups)
        k_1x1_avg_second, b_1x1_avg_second = transI_fusebn(k_avg,
                                                           self.dbb_avg.avgbn)
        if hasattr(self.dbb_avg, 'conv'):
            k_1x1_avg_first, b_1x1_avg_first = transI_fusebn(
                self.dbb_avg.conv.weight, self.dbb_avg.bn)
            k_1x1_avg_merged, b_1x1_avg_merged = transIII_1x1_kxk(
                k_1x1_avg_first,
                b_1x1_avg_first,
                k_1x1_avg_second,
                b_1x1_avg_second,
                groups=self.groups)
        else:
            k_1x1_avg_merged, b_1x1_avg_merged = k_1x1_avg_second, b_1x1_avg_second

        return transII_addbranch(
            (k_origin, k_1x1, k_1x1_kxk_merged, k_1x1_avg_merged),
            (b_origin, b_1x1, b_1x1_kxk_merged, b_1x1_avg_merged))

    def re_parameterize(self):
        if self.is_repped:
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.dbb_reparam = nn.Conv2D(
            in_channels=self.dbb_origin.conv._in_channels,
            out_channels=self.dbb_origin.conv._out_channels,
            kernel_size=self.dbb_origin.conv._kernel_size,
            stride=self.dbb_origin.conv._stride,
            padding=self.dbb_origin.conv._padding,
            dilation=self.dbb_origin.conv._dilation,
            groups=self.dbb_origin.conv._groups,
            bias_attr=True)

        self.dbb_reparam.weight.set_value(kernel)
        self.dbb_reparam.bias.set_value(bias)

        self.__delattr__('dbb_origin')
        self.__delattr__('dbb_avg')
        if hasattr(self, 'dbb_1x1'):
            self.__delattr__('dbb_1x1')
        self.__delattr__('dbb_1x1_kxk')
        self.is_repped = True
