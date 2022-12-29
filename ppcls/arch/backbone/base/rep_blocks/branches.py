import paddle
from paddle import nn
import paddle.nn.functional as F

from ..theseus_layer import TheseusLayer


class RepBranch(TheseusLayer):
    def __init__(self):
        super().__init__()

    def get_rep_kernel(self):
        raise NotImplementedError("")


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
                 conv=nn.Conv2D):
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
        if not isinstance(self.bn, nn.BatchNorm2D):
            kernel_hat = self.conv.kernel
            bias_hat = self.conv.bias
        else:
            gamma = self.bn.weight
            std = (self.bn._variance + self.bn._epsilon).sqrt()
            bias = -self.bn._mean
            if self.conv.bias is not None:
                bias += self.conv.bias

            kernel_hat = self.conv.kernel * (
                (gamma / std).reshape([-1, 1, 1, 1]))
            bias_hat = self.bn.bias + bias * gamma / std
        return kernel_hat, bias_hat


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
    def __init__(self, in_channels, out_channels, stride, groups,
                 with_bn=True):
        super().__init__()
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
            self.block = None

    def forward(self, x):
        if self.block:
            return self.block(x)
        else:
            return 0

    def get_rep_kernel(self):
        if self.block:
            return self.block.get_rep_kernel()
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
            if internal_channels == in_channels else nn.Conv2D)
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
        k1, b1 = self.conv1x1.get_rep_kernel()
        k2, b2 = self.convkxk.get_rep_kernel()

        if self.groups == 1:
            kernel_hat = F.conv2d(k2, k1.transpose([1, 0, 2, 3]))
            bias_hat = (k2 * b1.reshape([1, -1, 1, 1])).sum((1, 2, 3)) + b2
        else:
            k_slices = []
            b_slices = []
            k1_T = k1.transpose([1, 0, 2, 3])
            k1_group_width = k1.shape[0] // groups
            k2_group_width = k2.shape[0] // groups
            for g in range(groups):
                k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) *
                                  k1_group_width, :, :]
                k2_slice = k2[g * k2_group_width:(g + 1) *
                              k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append((k2_slice * b1[g * k1_group_width:(
                    g + 1) * k1_group_width].reshape([1, -1, 1, 1])).sum((1, 2,
                                                                          3)))
            kernel_hat = paddle.concat(k_slices)
            bias_hat = paddle.concat(b_slices) + b2
        return kernel_hat, bias_hat


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
        pass
