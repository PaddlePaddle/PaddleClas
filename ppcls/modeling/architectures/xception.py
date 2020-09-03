import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
import math

__all__ = ['Xception41', 'Xception65', 'Xception71']


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 num_channels,
                 num_filters,
                 filter_size,
                 stride=1,
                 groups=1,
                 act=None,
                 name=None):
        super(ConvBNLayer, self).__init__()

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
        bn_name = "bn_" + name
        self._batch_norm = BatchNorm(
            num_filters,
            act=act,
            param_attr=ParamAttr(name=bn_name + "_scale"),
            bias_attr=ParamAttr(name=bn_name + "_offset"),
            moving_mean_name=bn_name + '_mean',
            moving_variance_name=bn_name + '_variance')

    def forward(self, inputs):
        y = self._conv(inputs)
        y = self._batch_norm(y)
        return y


class SeparableConv(fluid.dygraph.Layer):
    def __init__(self, input_channels, output_channels, stride=1, name=None):
        super(SeparableConv, self).__init__()

        self._pointwise_conv = ConvBNLayer(
            input_channels, output_channels, 1, name=name + "_sep")
        self._depthwise_conv = ConvBNLayer(
            output_channels,
            output_channels,
            3,
            stride=stride,
            groups=output_channels,
            name=name + "_dw")

    def forward(self, inputs):
        x = self._pointwise_conv(inputs)
        x = self._depthwise_conv(x)
        return x


class EntryFlowBottleneckBlock(fluid.dygraph.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 stride=2,
                 name=None,
                 relu_first=False):
        super(EntryFlowBottleneckBlock, self).__init__()
        self.relu_first = relu_first

        self._short = Conv2D(
            num_channels=input_channels,
            num_filters=output_channels,
            filter_size=1,
            stride=stride,
            padding=0,
            act=None,
            param_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)
        self._conv1 = SeparableConv(
            input_channels,
            output_channels,
            stride=1,
            name=name + "_branch2a_weights")
        self._conv2 = SeparableConv(
            output_channels,
            output_channels,
            stride=1,
            name=name + "_branch2b_weights")
        self._pool = Pool2D(
            pool_size=3, pool_stride=stride, pool_padding=1, pool_type="max")

    def forward(self, inputs):
        conv0 = inputs
        short = self._short(inputs)
        if self.relu_first:
            conv0 = fluid.layers.relu(conv0)
        conv1 = self._conv1(conv0)
        conv2 = fluid.layers.relu(conv1)
        conv2 = self._conv2(conv2)
        pool = self._pool(conv2)
        return fluid.layers.elementwise_add(x=short, y=pool)


class EntryFlow(fluid.dygraph.Layer):
    def __init__(self, block_num=3):
        super(EntryFlow, self).__init__()

        name = "entry_flow"
        self.block_num = block_num
        self._conv1 = ConvBNLayer(
            3, 32, 3, stride=2, act="relu", name=name + "_conv1")
        self._conv2 = ConvBNLayer(32, 64, 3, act="relu", name=name + "_conv2")
        if block_num == 3:
            self._conv_0 = EntryFlowBottleneckBlock(
                64, 128, stride=2, name=name + "_0", relu_first=False)
            self._conv_1 = EntryFlowBottleneckBlock(
                128, 256, stride=2, name=name + "_1", relu_first=True)
            self._conv_2 = EntryFlowBottleneckBlock(
                256, 728, stride=2, name=name + "_2", relu_first=True)
        elif block_num == 5:
            self._conv_0 = EntryFlowBottleneckBlock(
                64, 128, stride=2, name=name + "_0", relu_first=False)
            self._conv_1 = EntryFlowBottleneckBlock(
                128, 256, stride=1, name=name + "_1", relu_first=True)
            self._conv_2 = EntryFlowBottleneckBlock(
                256, 256, stride=2, name=name + "_2", relu_first=True)
            self._conv_3 = EntryFlowBottleneckBlock(
                256, 728, stride=1, name=name + "_3", relu_first=True)
            self._conv_4 = EntryFlowBottleneckBlock(
                728, 728, stride=2, name=name + "_4", relu_first=True)
        else:
            sys.exit(-1)

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)

        if self.block_num == 3:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
        elif self.block_num == 5:
            x = self._conv_0(x)
            x = self._conv_1(x)
            x = self._conv_2(x)
            x = self._conv_3(x)
            x = self._conv_4(x)
        return x


class MiddleFlowBottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, input_channels, output_channels, name):
        super(MiddleFlowBottleneckBlock, self).__init__()

        self._conv_0 = SeparableConv(
            input_channels,
            output_channels,
            stride=1,
            name=name + "_branch2a_weights")
        self._conv_1 = SeparableConv(
            output_channels,
            output_channels,
            stride=1,
            name=name + "_branch2b_weights")
        self._conv_2 = SeparableConv(
            output_channels,
            output_channels,
            stride=1,
            name=name + "_branch2c_weights")

    def forward(self, inputs):
        conv0 = fluid.layers.relu(inputs)
        conv0 = self._conv_0(conv0)
        conv1 = fluid.layers.relu(conv0)
        conv1 = self._conv_1(conv1)
        conv2 = fluid.layers.relu(conv1)
        conv2 = self._conv_2(conv2)
        return fluid.layers.elementwise_add(x=inputs, y=conv2)


class MiddleFlow(fluid.dygraph.Layer):
    def __init__(self, block_num=8):
        super(MiddleFlow, self).__init__()

        self.block_num = block_num
        self._conv_0 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_0")
        self._conv_1 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_1")
        self._conv_2 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_2")
        self._conv_3 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_3")
        self._conv_4 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_4")
        self._conv_5 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_5")
        self._conv_6 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_6")
        self._conv_7 = MiddleFlowBottleneckBlock(
            728, 728, name="middle_flow_7")
        if block_num == 16:
            self._conv_8 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_8")
            self._conv_9 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_9")
            self._conv_10 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_10")
            self._conv_11 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_11")
            self._conv_12 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_12")
            self._conv_13 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_13")
            self._conv_14 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_14")
            self._conv_15 = MiddleFlowBottleneckBlock(
                728, 728, name="middle_flow_15")

    def forward(self, inputs):
        x = self._conv_0(inputs)
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        x = self._conv_4(x)
        x = self._conv_5(x)
        x = self._conv_6(x)
        x = self._conv_7(x)
        if self.block_num == 16:
            x = self._conv_8(x)
            x = self._conv_9(x)
            x = self._conv_10(x)
            x = self._conv_11(x)
            x = self._conv_12(x)
            x = self._conv_13(x)
            x = self._conv_14(x)
            x = self._conv_15(x)
        return x


class ExitFlowBottleneckBlock(fluid.dygraph.Layer):
    def __init__(self, input_channels, output_channels1, output_channels2,
                 name):
        super(ExitFlowBottleneckBlock, self).__init__()

        self._short = Conv2D(
            num_channels=input_channels,
            num_filters=output_channels2,
            filter_size=1,
            stride=2,
            padding=0,
            act=None,
            param_attr=ParamAttr(name + "_branch1_weights"),
            bias_attr=False)
        self._conv_1 = SeparableConv(
            input_channels,
            output_channels1,
            stride=1,
            name=name + "_branch2a_weights")
        self._conv_2 = SeparableConv(
            output_channels1,
            output_channels2,
            stride=1,
            name=name + "_branch2b_weights")
        self._pool = Pool2D(
            pool_size=3, pool_stride=2, pool_padding=1, pool_type="max")

    def forward(self, inputs):
        short = self._short(inputs)
        conv0 = fluid.layers.relu(inputs)
        conv1 = self._conv_1(conv0)
        conv2 = fluid.layers.relu(conv1)
        conv2 = self._conv_2(conv2)
        pool = self._pool(conv2)
        return fluid.layers.elementwise_add(x=short, y=pool)


class ExitFlow(fluid.dygraph.Layer):
    def __init__(self, class_dim):
        super(ExitFlow, self).__init__()

        name = "exit_flow"

        self._conv_0 = ExitFlowBottleneckBlock(
            728, 728, 1024, name=name + "_1")
        self._conv_1 = SeparableConv(1024, 1536, stride=1, name=name + "_2")
        self._conv_2 = SeparableConv(1536, 2048, stride=1, name=name + "_3")
        self._pool = Pool2D(pool_type="avg", global_pooling=True)
        stdv = 1.0 / math.sqrt(2048 * 1.0)
        self._out = Linear(
            2048,
            class_dim,
            param_attr=ParamAttr(
                name="fc_weights",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs):
        conv0 = self._conv_0(inputs)
        conv1 = self._conv_1(conv0)
        conv1 = fluid.layers.relu(conv1)
        conv2 = self._conv_2(conv1)
        conv2 = fluid.layers.relu(conv2)
        pool = self._pool(conv2)
        pool = fluid.layers.reshape(pool, [0, -1])
        out = self._out(pool)
        return out


class Xception(fluid.dygraph.Layer):
    def __init__(self,
                 entry_flow_block_num=3,
                 middle_flow_block_num=8,
                 class_dim=1000):
        super(Xception, self).__init__()
        self.entry_flow_block_num = entry_flow_block_num
        self.middle_flow_block_num = middle_flow_block_num
        self._entry_flow = EntryFlow(entry_flow_block_num)
        self._middle_flow = MiddleFlow(middle_flow_block_num)
        self._exit_flow = ExitFlow(class_dim)

    def forward(self, inputs):
        x = self._entry_flow(inputs)
        x = self._middle_flow(x)
        x = self._exit_flow(x)
        return x


def Xception41(**args):
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=8, **args)
    return model


def Xception65(**args):
    model = Xception(entry_flow_block_num=3, middle_flow_block_num=16, **args)
    return model


def Xception71(**args):
    model = Xception(entry_flow_block_num=5, middle_flow_block_num=16, **args)
    return model