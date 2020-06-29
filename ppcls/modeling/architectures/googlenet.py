import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
import math

__all__ = ['GoogLeNet_DY']

def xavier(channels, filter_size, name):
    stdv = (3.0 / (filter_size**2 * channels))**0.5
    param_attr = ParamAttr(
        initializer=fluid.initializer.Uniform(-stdv, stdv),
        name=name + "_weights")

    return param_attr


class ConvLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=num_channels,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(name=name + "_weights"),
            bias_attr=False)

    def forward(self, inputs):
        y = self._conv(inputs)
        return y


class Inception(fluid.dygraph.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter1,
                 filter3R,
                 filter3,
                 filter5R,
                 filter5,
                 proj,
                 name=None):
        super(Inception, self).__init__()

        self._conv1 = ConvLayer(
            input_channels, filter1, 1, name="inception_" + name + "_1x1")
        self._conv3r = ConvLayer(
            input_channels,
            filter3R,
            1,
            name="inception_" + name + "_3x3_reduce")
        self._conv3 = ConvLayer(
            filter3R, filter3, 3, name="inception_" + name + "_3x3")
        self._conv5r = ConvLayer(
            input_channels,
            filter5R,
            1,
            name="inception_" + name + "_5x5_reduce")
        self._conv5 = ConvLayer(
            filter5R, filter5, 5, name="inception_" + name + "_5x5")
        self._pool = Pool2D(
            pool_size=3, pool_type="max", pool_stride=1, pool_padding=1)
        self._convprj = ConvLayer(
            input_channels, proj, 1, name="inception_" + name + "_3x3_proj")

    def forward(self, inputs):
        conv1 = self._conv1(inputs)

        conv3r = self._conv3r(inputs)
        conv3 = self._conv3(conv3r)

        conv5r = self._conv5r(inputs)
        conv5 = self._conv5(conv5r)

        pool = self._pool(inputs)
        convprj = self._convprj(pool)

        cat = fluid.layers.concat([conv1, conv3, conv5, convprj], axis=1)
        layer_helper = LayerHelper(self.full_name(), act="relu")
        return layer_helper.append_activation(cat)


class GoogleNetDY(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000):
        super(GoogleNetDY, self).__init__()
        self._conv = ConvLayer(3, 64, 7, 2, name="conv1")
        self._pool = Pool2D(pool_size=3, pool_type="max", pool_stride=2)
        self._conv_1 = ConvLayer(64, 64, 1, name="conv2_1x1")
        self._conv_2 = ConvLayer(64, 192, 3, name="conv2_3x3")

        self._ince3a = Inception(
            192, 192, 64, 96, 128, 16, 32, 32, name="ince3a")
        self._ince3b = Inception(
            256, 256, 128, 128, 192, 32, 96, 64, name="ince3b")

        self._ince4a = Inception(
            480, 480, 192, 96, 208, 16, 48, 64, name="ince4a")
        self._ince4b = Inception(
            512, 512, 160, 112, 224, 24, 64, 64, name="ince4b")
        self._ince4c = Inception(
            512, 512, 128, 128, 256, 24, 64, 64, name="ince4c")
        self._ince4d = Inception(
            512, 512, 112, 144, 288, 32, 64, 64, name="ince4d")
        self._ince4e = Inception(
            528, 528, 256, 160, 320, 32, 128, 128, name="ince4e")

        self._ince5a = Inception(
            832, 832, 256, 160, 320, 32, 128, 128, name="ince5a")
        self._ince5b = Inception(
            832, 832, 384, 192, 384, 48, 128, 128, name="ince5b")

        self._pool_5 = Pool2D(pool_size=7, pool_type='avg', pool_stride=7)

        self._drop = fluid.dygraph.Dropout(p=0.4)
        self._fc_out = Linear(
            1024,
            class_dim,
            param_attr=xavier(1024, 1, "out"),
            bias_attr=ParamAttr(name="out_offset"),
            act="softmax")
        self._pool_o1 = Pool2D(pool_size=5, pool_stride=3, pool_type="avg")
        self._conv_o1 = ConvLayer(512, 128, 1, name="conv_o1")
        self._fc_o1 = Linear(
            1152,
            1024,
            param_attr=xavier(2048, 1, "fc_o1"),
            bias_attr=ParamAttr(name="fc_o1_offset"),
            act="relu")
        self._drop_o1 = fluid.dygraph.Dropout(p=0.7)
        self._out1 = Linear(
            1024,
            class_dim,
            param_attr=xavier(1024, 1, "out1"),
            bias_attr=ParamAttr(name="out1_offset"),
            act="softmax")
        self._pool_o2 = Pool2D(pool_size=5, pool_stride=3, pool_type='avg')
        self._conv_o2 = ConvLayer(528, 128, 1, name="conv_o2")
        self._fc_o2 = Linear(
            1152,
            1024,
            param_attr=xavier(2048, 1, "fc_o2"),
            bias_attr=ParamAttr(name="fc_o2_offset"))
        self._drop_o2 = fluid.dygraph.Dropout(p=0.7)
        self._out2 = Linear(
            1024,
            class_dim,
            param_attr=xavier(1024, 1, "out2"),
            bias_attr=ParamAttr(name="out2_offset"))

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._pool(x)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._pool(x)

        x = self._ince3a(x)
        x = self._ince3b(x)
        x = self._pool(x)

        ince4a = self._ince4a(x)
        x = self._ince4b(ince4a)
        x = self._ince4c(x)
        ince4d = self._ince4d(x)
        x = self._ince4e(ince4d)
        x = self._pool(x)

        x = self._ince5a(x)
        ince5b = self._ince5b(x)

        x = self._pool_5(ince5b)
        x = self._drop(x)
        x = fluid.layers.squeeze(x, axes=[2, 3])
        out = self._fc_out(x)

        x = self._pool_o1(ince4a)
        x = self._conv_o1(x)
        x = fluid.layers.flatten(x)
        x = self._fc_o1(x)
        x = self._drop_o1(x)
        out1 = self._out1(x)

        x = self._pool_o2(ince4d)
        x = self._conv_o2(x)
        x = fluid.layers.flatten(x)
        x = self._fc_o2(x)
        x = self._drop_o2(x)
        out2 = self._out2(x)
        return [out, out1, out2]


def GoogLeNet(**args):
    model = GoogleNetDY(**args)
    return model