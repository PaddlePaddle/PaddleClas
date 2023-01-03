import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np


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
        # nn.init.zeros_(self.weight)
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
                 **kwargs):
        super().__init__()

        padding = (filter_size - 1) // 2
        dilation = 1
        deploy = False
        single_init = False
        in_channels = num_channels
        out_channels = num_filters
        kernel_size = filter_size
        internal_channels_1x1_3x3 = None
        nonlinear = act

        self.deploy = deploy

        if nonlinear is None:
            self.nonlinear = nn.Identity()
        else:
            self.nonlinear = nn.ReLU()

        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.groups = groups
        assert padding == kernel_size // 2

        if deploy:
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

        if hasattr(self, 'dbb_reparam'):
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
