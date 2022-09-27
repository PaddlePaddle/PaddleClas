# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
# this code is based on AdaFace(https://github.com/mk-minchul/AdaFace)
from collections import namedtuple
import paddle
import paddle.nn as nn
from paddle.nn import Dropout
from paddle.nn import MaxPool2D
from paddle.nn import Sequential
from paddle.nn import Conv2D, Linear
from paddle.nn import BatchNorm1D, BatchNorm2D
from paddle.nn import ReLU, Sigmoid
from paddle.nn import Layer
from paddle.nn import PReLU

# from ppcls.arch.backbone.legendary_models.resnet import _load_pretrained


class Flatten(Layer):
    """ Flat tensor
    """

    def forward(self, input):
        return paddle.reshape(input, [input.shape[0], -1])


class LinearBlock(Layer):
    """ Convolution block without no-linear activation layer
    """

    def __init__(self,
                 in_c,
                 out_c,
                 kernel=(1, 1),
                 stride=(1, 1),
                 padding=(0, 0),
                 groups=1):
        super(LinearBlock, self).__init__()
        self.conv = Conv2D(
            in_c,
            out_c,
            kernel,
            stride,
            padding,
            groups=groups,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=None)
        weight_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=0.0))
        self.bn = BatchNorm2D(
            out_c, weight_attr=weight_attr, bias_attr=bias_attr)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class GNAP(Layer):
    """ Global Norm-Aware Pooling block
    """

    def __init__(self, in_c):
        super(GNAP, self).__init__()
        self.bn1 = BatchNorm2D(in_c, weight_attr=False, bias_attr=False)
        self.pool = nn.AdaptiveAvgPool2D((1, 1))
        self.bn2 = BatchNorm1D(in_c, weight_attr=False, bias_attr=False)

    def forward(self, x):
        x = self.bn1(x)
        x_norm = paddle.norm(x, 2, 1, True)
        x_norm_mean = paddle.mean(x_norm)
        weight = x_norm_mean / x_norm
        x = x * weight
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        feature = self.bn2(x)
        return feature


class GDC(Layer):
    """ Global Depthwise Convolution block
    """

    def __init__(self, in_c, embedding_size):
        super(GDC, self).__init__()
        self.conv_6_dw = LinearBlock(
            in_c,
            in_c,
            groups=in_c,
            kernel=(7, 7),
            stride=(1, 1),
            padding=(0, 0))
        self.conv_6_flatten = Flatten()
        self.linear = Linear(
            in_c,
            embedding_size,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=False)
        self.bn = BatchNorm1D(
            embedding_size, weight_attr=False, bias_attr=False)

    def forward(self, x):
        x = self.conv_6_dw(x)
        x = self.conv_6_flatten(x)
        x = self.linear(x)
        x = self.bn(x)
        return x


class SELayer(Layer):
    """ SE block
    """

    def __init__(self, channels, reduction):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        weight_attr = paddle.ParamAttr(
            initializer=paddle.nn.initializer.XavierUniform())
        self.fc1 = Conv2D(
            channels,
            channels // reduction,
            kernel_size=1,
            padding=0,
            weight_attr=weight_attr,
            bias_attr=False)

        self.relu = ReLU()
        self.fc2 = Conv2D(
            channels // reduction,
            channels,
            kernel_size=1,
            padding=0,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=False)

        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return module_input * x


class BasicBlockIR(Layer):
    """ BasicBlock for IRNet
    """

    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIR, self).__init__()

        weight_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=0.0))
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(
                    in_channel,
                    depth, (1, 1),
                    stride,
                    weight_attr=nn.initializer.KaimingNormal(),
                    bias_attr=False),
                BatchNorm2D(
                    depth, weight_attr=weight_attr, bias_attr=bias_attr))
        self.res_layer = Sequential(
            BatchNorm2D(
                in_channel, weight_attr=weight_attr, bias_attr=bias_attr),
            Conv2D(
                in_channel,
                depth, (3, 3), (1, 1),
                1,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                depth, weight_attr=weight_attr, bias_attr=bias_attr),
            PReLU(depth),
            Conv2D(
                depth,
                depth, (3, 3),
                stride,
                1,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                depth, weight_attr=weight_attr, bias_attr=bias_attr))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BottleneckIR(Layer):
    """ BasicBlock with bottleneck for IRNet
    """

    def __init__(self, in_channel, depth, stride):
        super(BottleneckIR, self).__init__()
        reduction_channel = depth // 4
        weight_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=0.0))
        if in_channel == depth:
            self.shortcut_layer = MaxPool2D(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2D(
                    in_channel,
                    depth, (1, 1),
                    stride,
                    weight_attr=nn.initializer.KaimingNormal(),
                    bias_attr=False),
                BatchNorm2D(
                    depth, weight_attr=weight_attr, bias_attr=bias_attr))
        self.res_layer = Sequential(
            BatchNorm2D(
                in_channel, weight_attr=weight_attr, bias_attr=bias_attr),
            Conv2D(
                in_channel,
                reduction_channel, (1, 1), (1, 1),
                0,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                reduction_channel,
                weight_attr=weight_attr,
                bias_attr=bias_attr),
            PReLU(reduction_channel),
            Conv2D(
                reduction_channel,
                reduction_channel, (3, 3), (1, 1),
                1,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                reduction_channel,
                weight_attr=weight_attr,
                bias_attr=bias_attr),
            PReLU(reduction_channel),
            Conv2D(
                reduction_channel,
                depth, (1, 1),
                stride,
                0,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                depth, weight_attr=weight_attr, bias_attr=bias_attr))

    def forward(self, x):
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)

        return res + shortcut


class BasicBlockIRSE(BasicBlockIR):
    def __init__(self, in_channel, depth, stride):
        super(BasicBlockIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_sublayer("se_block", SELayer(depth, 16))


class BottleneckIRSE(BottleneckIR):
    def __init__(self, in_channel, depth, stride):
        super(BottleneckIRSE, self).__init__(in_channel, depth, stride)
        self.res_layer.add_sublayer("se_block", SELayer(depth, 16))


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    '''A named tuple describing a ResNet block.'''


def get_block(in_channel, depth, num_units, stride=2):

    return [Bottleneck(in_channel, depth, stride)] +\
           [Bottleneck(depth, depth, 1) for i in range(num_units - 1)]


def get_blocks(num_layers):
    if num_layers == 18:
        blocks = [
            get_block(
                in_channel=64, depth=64, num_units=2), get_block(
                    in_channel=64, depth=128, num_units=2), get_block(
                        in_channel=128, depth=256, num_units=2), get_block(
                            in_channel=256, depth=512, num_units=2)
        ]
    elif num_layers == 34:
        blocks = [
            get_block(
                in_channel=64, depth=64, num_units=3), get_block(
                    in_channel=64, depth=128, num_units=4), get_block(
                        in_channel=128, depth=256, num_units=6), get_block(
                            in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 50:
        blocks = [
            get_block(
                in_channel=64, depth=64, num_units=3), get_block(
                    in_channel=64, depth=128, num_units=4), get_block(
                        in_channel=128, depth=256, num_units=14), get_block(
                            in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 100:
        blocks = [
            get_block(
                in_channel=64, depth=64, num_units=3), get_block(
                    in_channel=64, depth=128, num_units=13), get_block(
                        in_channel=128, depth=256, num_units=30), get_block(
                            in_channel=256, depth=512, num_units=3)
        ]
    elif num_layers == 152:
        blocks = [
            get_block(
                in_channel=64, depth=256, num_units=3), get_block(
                    in_channel=256, depth=512, num_units=8), get_block(
                        in_channel=512, depth=1024, num_units=36), get_block(
                            in_channel=1024, depth=2048, num_units=3)
        ]
    elif num_layers == 200:
        blocks = [
            get_block(
                in_channel=64, depth=256, num_units=3), get_block(
                    in_channel=256, depth=512, num_units=24), get_block(
                        in_channel=512, depth=1024, num_units=36), get_block(
                            in_channel=1024, depth=2048, num_units=3)
        ]

    return blocks


class Backbone(Layer):
    def __init__(self, input_size, num_layers, mode='ir'):
        """ Args:
            input_size: input_size of backbone
            num_layers: num_layers of backbone
            mode: support ir or irse
        """
        super(Backbone, self).__init__()
        assert input_size[0] in [112, 224], \
            "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], \
            "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ['ir', 'ir_se'], \
            "mode should be ir or ir_se"
        weight_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=1.0))
        bias_attr = paddle.ParamAttr(
            regularizer=None, initializer=nn.initializer.Constant(value=0.0))
        self.input_layer = Sequential(
            Conv2D(
                3,
                64, (3, 3),
                1,
                1,
                weight_attr=nn.initializer.KaimingNormal(),
                bias_attr=False),
            BatchNorm2D(
                64, weight_attr=weight_attr, bias_attr=bias_attr),
            PReLU(64))
        blocks = get_blocks(num_layers)
        if num_layers <= 100:
            if mode == 'ir':
                unit_module = BasicBlockIR
            elif mode == 'ir_se':
                unit_module = BasicBlockIRSE
            output_channel = 512
        else:
            if mode == 'ir':
                unit_module = BottleneckIR
            elif mode == 'ir_se':
                unit_module = BottleneckIRSE
            output_channel = 2048

        if input_size[0] == 112:
            self.output_layer = Sequential(
                BatchNorm2D(
                    output_channel,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr),
                Dropout(0.4),
                Flatten(),
                Linear(
                    output_channel * 7 * 7,
                    512,
                    weight_attr=nn.initializer.KaimingNormal()),
                BatchNorm1D(
                    512, weight_attr=False, bias_attr=False))
        else:
            self.output_layer = Sequential(
                BatchNorm2D(
                    output_channel,
                    weight_attr=weight_attr,
                    bias_attr=bias_attr),
                Dropout(0.4),
                Flatten(),
                Linear(
                    output_channel * 14 * 14,
                    512,
                    weight_attr=nn.initializer.KaimingNormal()),
                BatchNorm1D(
                    512, weight_attr=False, bias_attr=False))

        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    unit_module(bottleneck.in_channel, bottleneck.depth,
                                bottleneck.stride))
        self.body = Sequential(*modules)

        # initialize_weights(self.modules())

    def forward(self, x):

        # current code only supports one extra image
        # it comes with a extra dimension for number of extra image. We will just squeeze it out for now
        x = self.input_layer(x)

        for idx, module in enumerate(self.body):
            x = module(x)

        x = self.output_layer(x)
        # norm = paddle.norm(x, 2, 1, True)
        # output = paddle.divide(x, norm)
        # return output, norm
        return x


def AdaFace_IR_18(input_size=(112, 112)):
    """ Constructs a ir-18 model.
    """
    model = Backbone(input_size, 18, 'ir')
    return model


def AdaFace_IR_34(input_size=(112, 112)):
    """ Constructs a ir-34 model.
    """
    model = Backbone(input_size, 34, 'ir')

    return model


def AdaFace_IR_50(input_size=(112, 112)):
    """ Constructs a ir-50 model.
    """
    model = Backbone(input_size, 50, 'ir')

    return model


def AdaFace_IR_101(input_size=(112, 112)):
    """ Constructs a ir-101 model.
    """
    model = Backbone(input_size, 100, 'ir')

    return model


def AdaFace_IR_152(input_size=(112, 112)):
    """ Constructs a ir-152 model.
    """
    model = Backbone(input_size, 152, 'ir')

    return model


def AdaFace_IR_200(input_size=(112, 112)):
    """ Constructs a ir-200 model.
    """
    model = Backbone(input_size, 200, 'ir')

    return model


def AdaFace_IR_SE_50(input_size=(112, 112)):
    """ Constructs a ir_se-50 model.
    """
    model = Backbone(input_size, 50, 'ir_se')

    return model


def AdaFace_IR_SE_101(input_size=(112, 112)):
    """ Constructs a ir_se-101 model.
    """
    model = Backbone(input_size, 100, 'ir_se')

    return model


def AdaFace_IR_SE_152(input_size=(112, 112)):
    """ Constructs a ir_se-152 model.
    """
    model = Backbone(input_size, 152, 'ir_se')

    return model


def AdaFace_IR_SE_200(input_size=(112, 112)):
    """ Constructs a ir_se-200 model.
    """
    model = Backbone(input_size, 200, 'ir_se')

    return model
