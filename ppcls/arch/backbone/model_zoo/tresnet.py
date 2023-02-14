import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from functools import partial
import os
import numpy as np


# Calculate symmetric padding for a convolution
def get_padding(kernel_size: int, stride: int = 1, dilation: int = 1, **_) -> int:
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


class BlurPool2d(nn.Layer):
    r"""Creates a module that computes blurs and downsample a given feature map.
    See :cite:`zhang2019shiftinvar` for more details.
    Corresponds to the Downsample class, which does blurring and subsampling
    Args:
        channels = Number of input channels
        filt_size (int): binomial filter size for blurring. currently supports 3 (default) and 5.
        stride (int): downsampling filter stride
    Returns:
        torch.Tensor: the transformed tensor.
    """
    def __init__(self, channels, filt_size=3, stride=2) -> None:
        super(BlurPool2d, self).__init__()
        assert filt_size > 1
        self.channels = channels
        self.filt_size = filt_size
        self.stride = stride
        self.padding = [get_padding(filt_size, stride, dilation=1)] * 4
        coeffs = paddle.to_tensor((np.poly1d((0.5, 0.5)) ** (self.filt_size - 1)).coeffs.astype(np.float32))
        blur_filter = (coeffs[:, None] * coeffs[None, :])[None, None, :, :].tile([self.channels, 1, 1, 1])
        self.register_buffer('filt', blur_filter, persistable=False)

    def forward(self, x):
        x = F.pad(x, self.padding, 'reflect')
        return F.conv2d(x, self.filt, stride=self.stride, groups=self.channels)


class AntiAliasDownsampleLayer(nn.Layer):
    def __init__(self, filt_size: int = 3, stride: int = 2,
                 channels: int = 0):
        super(AntiAliasDownsampleLayer, self).__init__()
        self.op = Downsample(filt_size, stride, channels)

    def forward(self, x):
        return self.op(x)

class Downsample(nn.Layer):
    def __init__(self, filt_size=3, stride=2, channels=None):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.stride = stride
        self.channels = channels

        assert self.filt_size == 3
        a = paddle.to_tensor([1., 2., 1.])
        filt = (a[:, None] * a[None, :])
        filt = filt / paddle.sum(filt)
        self.filt = paddle.tile(filt[None, None, :, :], repeat_times=[self.channels, 1, 1, 1])
        self.register_buffer('filt', self.filt)

    def forward(self, input):
        input_pad = F.pad(input, (1, 1, 1, 1), 'reflect')
        return F.conv2d(input_pad, self.filt, stride=self.stride, padding=0, groups=input.shape[1])


class SpaceToDepth(nn.Layer):
    def __init__(self, block_size=4):
        super(SpaceToDepth, self).__init__()
        assert block_size == 4
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.shape
        x = paddle.reshape(x, [N, C, H // self.bs, self.bs, W // self.bs, self.bs])   # (N, C, H // bs, bs, W // bs, bs)
        x = paddle.transpose(x, [0, 3, 5, 1, 2, 4])   # (N, bs, bs, C, H // bs, W // bs)
        x = paddle.reshape(x, [N, self.bs * self.bs * C, H // self.bs, W // self.bs])
        return x


class SpaceToDepthModule(nn.Layer):
    def __init__(self):
        super(SpaceToDepthModule, self).__init__()
        self.op = SpaceToDepth()

    def forward(self, x):
        x = self.op(x)
        return x

class FastGlobalAvgPool2d(nn.Layer):
    def __init__(self, flatten=False):
        super(FastGlobalAvgPool2d, self).__init__()
        self.flatten = flatten

    def forward(self, x):
        in_size = x.shape
        if self.flatten:
            return paddle.reshape(x, [in_size[0], in_size[1], -1]).mean(axis=2)
        else:
            x = paddle.reshape(x, [in_size[0], in_size[1], -1])
            x = x.mean(-1)
            x = paddle.reshape(x, [in_size[0], in_size[1], 1, 1])
            return x

class SEModule(nn.Layer):
    def __init__(self, channels, reduction_channels, inplace=True):
        super(SEModule, self).__init__()
        self.avg_pool = FastGlobalAvgPool2d()
        self.fc1 = nn.Conv2D(channels, reduction_channels, kernel_size=1, padding=0, bias_attr=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2D(reduction_channels, channels, kernel_size=1, padding=0, bias_attr=True)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se2 = self.fc1(x_se)
        x_se2 = self.relu(x_se2)
        x_se = self.fc2(x_se2)
        x_se = self.activation(x_se)
        return x * x_se

class Conv2d_ABN(nn.Layer):
    def __init__(self, ni, nf, stride, activation='leaky_relu', kernel_size=3, activation_param=1e-2, groups=1):
        super(Conv2d_ABN, self).__init__()
        self.conv2d = nn.Conv2D(ni, nf, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, groups=groups,
                      bias_attr=False)
        self.bn = nn.BatchNorm2D(nf, momentum=0.1, use_global_stats=True)
        self.act = nn.LeakyReLU(negative_slope=activation_param) if activation == 'leaky_relu' else nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(BasicBlock, self).__init__()
        if stride == 1:
            self.conv1 = Conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv1 = Conv2d_ABN(inplanes, planes, stride=2, activation_param=1e-3)
            else:
                self.conv1 = nn.Sequential(Conv2d_ABN(inplanes, planes, stride=1, activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv2 = Conv2d_ABN(planes, planes, stride=1, activation='identity')
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        reduce_layer_planes = max(planes * self.expansion // 4, 64)
        self.se = SEModule(channels=planes * self.expansion, reduction_channels=reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.se is not None:
            out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True, anti_alias_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d_ABN(inplanes, planes, kernel_size=1, stride=1, activation='leaky_relu',
                                activation_param=1e-3)
        if stride == 1:
            self.conv2 = Conv2d_ABN(planes, planes, kernel_size=3, stride=1, activation='leaky_relu',
                                    activation_param=1e-3)
        else:
            if anti_alias_layer is None:
                self.conv2 = Conv2d_ABN(planes, planes, kernel_size=3, stride=2, activation='leaky_relu', activation_param=1e-3)
            else:
                self.conv2 = nn.Sequential(Conv2d_ABN(planes, planes, stride=1, activation='leaky_relu',
                                                      activation_param=1e-3),
                                           anti_alias_layer(channels=planes, filt_size=3, stride=2))
        self.conv3 = Conv2d_ABN(planes, planes * self.expansion, kernel_size=1, stride=1, activation='identity')
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

        reduce_layer_planes = max(planes * self.expansion // 8, 64)
        self.se = SEModule(planes, reduce_layer_planes) if use_se else None

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.se is not None: out = self.se(out)
        out = self.conv3(out)
        out += residual

        out = self.relu(out)

        return out

class TResNet(nn.Layer):
    def __init__(self, layers, in_chans=3, class_num=1000, width_factor=1.0, **kwargs):
        super(TResNet, self).__init__()
        self.inplanes = int(int(64 * width_factor + 4) / 8) * 8
        self.planes = int(int(64 * width_factor + 4) / 8) * 8
        SpaceToDepth = SpaceToDepthModule()
        conv1 = Conv2d_ABN(in_chans * 16, self.planes, stride=1, kernel_size=3)
        anti_alias_layer = partial(AntiAliasDownsampleLayer)
        global_pool_layer = FastGlobalAvgPool2d(flatten=True)
        layer1 = self._make_layer(BasicBlock, self.planes, layers[0], stride=1, use_se=True,
                                  anti_alias_layer=anti_alias_layer)   # 56x56
        layer2 = self._make_layer(BasicBlock, self.planes * 2, layers[1], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)   # 28x28
        layer3 = self._make_layer(Bottleneck, self.planes * 4, layers[2], stride=2, use_se=True,
                                  anti_alias_layer=anti_alias_layer)   # 14x14
        layer4 = self._make_layer(Bottleneck, self.planes * 8, layers[3], stride=2, use_se=False,
                                  anti_alias_layer=anti_alias_layer)   # 7x7

        self.body = nn.Sequential(
            ('SpaceToDepth', SpaceToDepth),
            ('conv1', conv1),
            ('layer1', layer1),
            ('layer2', layer2),
            ('layer3', layer3),
            ('layer4', layer4))

        # default head
        self.num_features = (self.planes * 8) * Bottleneck.expansion

        fc = nn.Linear(self.num_features, class_num, bias_attr=True)
        self.global_pool = nn.Sequential(
            ('global_pool_layer', global_pool_layer)
        )
        self.head = nn.Sequential(('fc', fc))
        self.embeddings = []

        # initialize
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                Kaiming_Normal = nn.initializer.KaimingNormal()
                Kaiming_Normal(m.weight)
            elif isinstance(m, nn.BatchNorm2D):
                Ones = nn.initializer.Constant(1.0)
                Zeors = nn.initializer.Constant(0.0)
                Ones(m.weight)
                Zeors(m.bias)

        for m in self.sublayers():
            if isinstance(m, BasicBlock):
                if isinstance(m, Conv2d_ABN):
                    if isinstance(m, nn.BatchNorm2D):
                        Zeors = nn.initializer.Constant(0.0)
                        Zeors(m.weight)

            if isinstance(m, Bottleneck):
                if isinstance(m, Conv2d_ABN):
                    if isinstance(m, nn.BatchNorm2D):
                        Zeors = nn.initializer.Constant(0.0)
                        Zeors(m.weight)

            if isinstance(m, nn.Linear):
                TrunctNormal = nn.initializer.TruncatedNormal(std=0.01)
                TrunctNormal(m.weight)

    def _make_layer(self, block, planes, blocks, stride=1, use_se=True,
                    anti_alias_layer=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            layers = []
            if stride == 2:
                # avg pooling before 1x1 conv
                layers.append(
                    nn.AvgPool2D(kernel_size=2, stride=2, ceil_mode=True))
            layers += [
                Conv2d_ABN(self.inplanes, planes * block.expansion, kernel_size=1, stride=1,
                           activation='identity')]
            downsample = nn.Sequential(*layers)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=use_se,
                            anti_alias_layer=anti_alias_layer))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes, use_se=use_se, anti_alias_layer=anti_alias_layer)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.body["SpaceToDepth"](x)
        x = self.body["conv1"](x)
        x = self.body["layer1"](x)
        x = self.body["layer2"](x)
        x = self.body["layer3"](x)
        x = self.body["layer4"](x)
        logits = self.head(self.global_pool(x))
        return logits

def load_dygraph_pretrain(model, path=None):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        raise ValueError("Model pretrain path {}.pdparams does not "
                         "exists.".format(path))
    param_state_dict = paddle.load(path + ".pdparams")
    if isinstance(model, list):
        for m in model:
            if hasattr(m, 'set_dict'):
                m.set_dict(param_state_dict)
    else:
        model.set_dict(param_state_dict)
    return

def _load_pretrained(pretrained, model, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        pass
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )

def TResNetM(pretrained=False, use_ssld=False, **kwargs):
    model = TResNet(layers=[3, 4, 11, 3], in_chans=3, width_factor=1.0, **kwargs)
    _load_pretrained(pretrained, model, use_ssld=use_ssld)
    return model


