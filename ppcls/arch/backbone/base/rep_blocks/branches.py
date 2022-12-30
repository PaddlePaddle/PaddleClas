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

import numpy as np
import paddle
from paddle import nn
import paddle.nn.functional as F

from ..theseus_layer import TheseusLayer


def fuse_bn_to_conv(weight, bias, bn):
    gamma = bn.weight
    std = (bn._variance + bn._epsilon).sqrt()
    bias_hat = -bn._mean
    if bias is not None:
        bias_hat += bias

    weight_hat = weight * ((gamma / std).reshape([-1, 1, 1, 1]))
    bias_hat = bn.bias + bias_hat * gamma / std
    return weight_hat, bias_hat


def fuse_conv1x1_to_convkxk(conv1x1_w, conv1x1_b, convkxk_w, convkxk_b,
                            groups):
    if groups == 1:
        weight_hat = F.conv2d(convkxk_w, conv1x1_w.transpose([1, 0, 2, 3]))
        bias_hat = (convkxk_w * conv1x1_b.reshape([1, -1, 1, 1])).sum(
            (1, 2, 3)) + convkxk_b
    else:
        k_slices = []
        b_slices = []
        conv1x1_w_T = conv1x1_w.transpose([1, 0, 2, 3])
        conv1x1_w_group_width = conv1x1_w.shape[0] // groups
        convkxk_w_group_width = convkxk_w.shape[0] // groups
        for g in range(groups):
            conv1x1_w_T_slice = conv1x1_w_T[:, g * conv1x1_w_group_width:(
                g + 1) * conv1x1_w_group_width, :, :]
            convkxk_w_slice = convkxk_w[g * convkxk_w_group_width:(g + 1) *
                                        convkxk_w_group_width, :, :, :]
            k_slices.append(F.conv2d(convkxk_w_slice, conv1x1_w_T_slice))
            b_slices.append((convkxk_w_slice * conv1x1_b[
                g * conv1x1_w_group_width:(g + 1) * conv1x1_w_group_width]
                             .reshape([1, -1, 1, 1])).sum((1, 2, 3)))
        weight_hat = paddle.concat(k_slices)
        bias_hat = paddle.concat(b_slices) + convkxk_b
    return weight_hat, bias_hat


class RepBranch(TheseusLayer):
    def __init__(self):
        super().__init__()

    def get_rep_kernel(self):
        raise NotImplementedError()


# TODO(gaotingquan): base nn.BatchNorm2D ?
class BNWithPad(nn.Layer):
    def __init__(self, num_features, padding=0, epsilon=1e-5, momentum=0.1):
        super().__init__()
        self.bn = nn.BatchNorm2D(
            num_features, momentum=momentum, epsilon=epsilon)
        self.padding = padding

    def forward(self, input):
        output = self.bn(input)
        if self.padding > 0:
            bias = -self.bn._mean
            pad_values = self.bn.bias + self.bn.weight * (
                bias / paddle.sqrt(self.bn._variance + self.bn._epsilon))
            ''' pad '''
            # TODO: n,h,w,c format is not supported yet
            n, c, h, w = output.shape
            values = pad_values.reshape([1, -1, 1, 1])
            w_values = values.expand([n, -1, self.padding, w])
            x = paddle.concat([w_values, output, w_values], axis=2)
            h = h + self.padding * 2
            h_values = values.expand([n, -1, h, self.padding])
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

    @property
    def _padding(self):
        return self.padding


class Conv2D(nn.Conv2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def equivalent_weight(self):
        return self.weight


class IdentityBasedConv(nn.Conv2D):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False):
        assert in_channels % groups == 0
        assert in_channels == out_channels

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias_attr)

        input_dim = in_channels // groups
        id_value = paddle.zeros(
            (in_channels, input_dim, kernel_size, kernel_size))
        for i in range(in_channels):
            id_value[i, i % input_dim, :, :] = 1
        self.id_tensor = id_value

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

    @property
    def equivalent_weight(self):
        return self.weight + self.id_tensor


# TODO(gaotingquan): support channel last format(NHWC) for resnet, etc.
class ConvBN(RepBranch):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False,
                 with_bn=True,
                 bn_padding=0,
                 conv=Conv2D):
        super().__init__()
        self.conv = conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)

        self.bn = BNWithPad(
            num_features=out_channels,
            padding=bn_padding) if with_bn else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

    def get_rep_kernel(self):
        if isinstance(self.bn, nn.Identity):
            weight_hat = self.conv.equivalent_weight
            bias_hat = self.conv.bias
        else:
            weight_hat, bias_hat = fuse_bn_to_conv(self.conv.equivalent_weight,
                                                   self.conv.bias, self.bn)
        return weight_hat, bias_hat


class ConvKxK(RepBranch):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 with_bn=True):
        super().__init__()
        self.block = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            with_bn=with_bn)

    def forward(self, x):
        return self.block(x)

    def get_rep_kernel(self):
        return self.block.get_rep_kernel()


class Conv1x1(RepBranch):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups,
                 with_bn=True,
                 to_rep_kernel_size=None):
        super().__init__()
        self.to_rep_kernel_size = to_rep_kernel_size

        if groups < out_channels:
            self.block = ConvBN(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                groups=groups,
                with_bn=with_bn)
        else:
            msg = f"The Conv1x1 can't be used when groups({groups} equal with out_channels({out_channels}))."
            logger.warning(msg)
            self.block = None

    def forward(self, x):
        if self.block:
            return self.block(x)
        else:
            return 0

    def get_rep_kernel(self):
        if self.block:
            if isinstance(self.to_rep_kernel_size, (list, tuple)):
                assert len(
                    self.to_rep_kernel_size
                ) == 2, f"The kernel_size's length must be 2. Received: {len(self.to_rep_kernel_size)}"
                assert self.to_rep_kernel_size[0] == self.to_rep_kernel_size[
                    1], f"The unsquare kernel is not supported. Received: {self.to_rep_kernel_size}"
                to_rep_kernel_size = self.to_rep_kernel_size[0]
            elif isinstance(self.to_rep_kernel_size, int):
                to_rep_kernel_size = self.to_rep_kernel_size
            else:
                msg = f"Required numerical type with '<class 'int'>', but received {type(self.to_rep_kernel_size)}."
                logger.error(msg)
                raise Exception(msg)
            assert to_rep_kernel_size % 2 == 1, "The even kernel size is not supported."

            w, b = self.block.get_rep_kernel()
            pad = (np.array(to_rep_kernel_size) - 1) // 2
            return F.pad(w, [pad, pad, pad, pad]), b
        else:
            return 0, 0


class Conv1x1_KxK(RepBranch):
    def __init__(self,
                 in_channels,
                 internal_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 with_bn=True):
        super().__init__()
        self.groups = groups
        self.conv1x1 = ConvBN(
            in_channels=in_channels,
            out_channels=internal_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            with_bn=with_bn,
            bn_padding=padding,
            conv=IdentityBasedConv
            if internal_channels == in_channels else Conv2D)
        self.convkxk = ConvBN(
            in_channels=internal_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            with_bn=with_bn)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.convkxk(x)
        return x

    def get_rep_kernel(self):
        conv1x1_w, conv1x1_b = self.conv1x1.get_rep_kernel()
        convkxk_w, convkxk_b = self.convkxk.get_rep_kernel()
        return fuse_conv1x1_to_convkxk(conv1x1_w, conv1x1_b, convkxk_w,
                                       convkxk_b, self.groups)


class Conv1x1_AVG(RepBranch):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 groups,
                 with_bn=True):
        super().__init__()
        self.out_channels, self.kernel_size, self.groups = out_channels, kernel_size, groups
        self.sequential = nn.Sequential()
        if groups < out_channels:
            self.sequential.add_sublayer(
                "conv",
                ConvBN(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=groups,
                    with_bn=with_bn,
                    bn_padding=padding))
            self.sequential.add_sublayer(
                "avg",
                nn.AvgPool2D(
                    kernel_size=kernel_size, stride=stride, padding=0))
        else:
            assert in_channels == out_channels
            self.sequential.add_sublayer(
                'avg',
                nn.AvgPool2D(
                    kernel_size=kernel_size, stride=stride, padding=padding))
        self.sequential.add_sublayer(
            'avgbn', nn.BatchNorm2D(num_features=out_channels))

    def forward(self, x):
        return self.sequential(x)

    def get_rep_kernel(self):
        # avg op
        input_dim = self.out_channels // self.groups
        avg_w = paddle.zeros([self.out_channels, input_dim] + self.kernel_size)
        avg_w[np.arange(self.out_channels), np.tile(
            np.arange(input_dim), self.groups), :, :] = 1.0 / (
                self.kernel_size[0] * self.kernel_size[1])

        # avgbn
        avg_w, avg_b = fuse_bn_to_conv(avg_w, 0, self.sequential["avgbn"])

        # conv op
        if hasattr(self.sequential, "conv"):
            conv1x1_w, conv1x1_b = self.sequential["conv"].get_rep_kernel()
            weight_hat, bias_hat = fuse_conv1x1_to_convkxk(
                conv1x1_w, conv1x1_b, avg_w, avg_b, self.groups)
        else:
            weight_hat, bias_hat = avg_w, avg_b

        return weight_hat, bias_hat
