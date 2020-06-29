import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
import math

__all__ = ["AlexNet"]

class ConvPoolLayer(fluid.dygraph.Layer):
    def __init__(self, 
                inputc_channels,
                output_channels,
                filter_size,
                stride,
                padding,
                stdv,
                groups=1,
                act=None,
                name=None):
        super(ConvPoolLayer, self).__init__()

        self._conv = Conv2D(num_channels=inputc_channels,
                            num_filters=output_channels,
                            filter_size=filter_size,
                            stride=stride,
                            padding=padding,
                            groups=groups,
                            param_attr=ParamAttr(name=name + "_weights",
                                initializer=fluid.initializer.Uniform(-stdv, stdv)),
                            bias_attr=ParamAttr(name=name + "_offset",
                                initializer=fluid.initializer.Uniform(-stdv, stdv)),
                            act=act)
        self._pool = Pool2D(pool_size=3,
                            pool_stride=2,
                            pool_padding=0,
                            pool_type="max")

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._pool(x)
        return x


class AlexNetDY(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000):
        super(AlexNetDY, self).__init__()

        stdv = 1.0/math.sqrt(3*11*11)
        self._conv1 = ConvPoolLayer(
            3, 64, 11, 4, 2, stdv, act="relu", name="conv1") 
        stdv = 1.0/math.sqrt(64*5*5)
        self._conv2 = ConvPoolLayer(
            64, 192, 5, 1, 2, stdv, act="relu", name="conv2")
        stdv = 1.0/math.sqrt(192*3*3)
        self._conv3 = Conv2D(192, 384, 3, stride=1, padding=1, 
            param_attr=ParamAttr(name="conv3_weights", initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="conv3_offset", initializer=fluid.initializer.Uniform(-stdv, stdv)), 
            act="relu")
        stdv = 1.0/math.sqrt(384*3*3)
        self._conv4 = Conv2D(384, 256, 3, stride=1, padding=1,
            param_attr=ParamAttr(name="conv4_weights", initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="conv4_offset", initializer=fluid.initializer.Uniform(-stdv, stdv)), 
            act="relu")
        stdv = 1.0/math.sqrt(256*3*3)
        self._conv5 = ConvPoolLayer(
            256, 256, 3, 1, 1, stdv, act="relu", name="conv5")
        stdv = 1.0/math.sqrt(256*6*6)

        self._drop1 = Dropout(p=0.5)
        self._fc6 = Linear(input_dim=256*6*6, 
                        output_dim=4096, 
                        param_attr=ParamAttr(name="fc6_weights", initializer=fluid.initializer.Uniform(-stdv, stdv)),
                        bias_attr=ParamAttr(name="fc6_offset", initializer=fluid.initializer.Uniform(-stdv, stdv)),
                        act="relu")
        
        self._drop2 = Dropout(p=0.5)
        self._fc7 = Linear(input_dim=4096,
                        output_dim=4096,
                        param_attr=ParamAttr(name="fc7_weights", initializer=fluid.initializer.Uniform(-stdv, stdv)),
                        bias_attr=ParamAttr(name="fc7_offset", initializer=fluid.initializer.Uniform(-stdv, stdv)),
                        act="relu")
        self._fc8 = Linear(input_dim=4096,
                        output_dim=class_dim,
                        param_attr=ParamAttr(name="fc8_weights", initializer=fluid.initializer.Uniform(-stdv, stdv)),
                        bias_attr=ParamAttr(name="fc8_offset", initializer=fluid.initializer.Uniform(-stdv, stdv)))

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        x = self._conv3(x)
        x = self._conv4(x)
        x = self._conv5(x)
        x = fluid.layers.flatten(x, axis=0)
        x = self._drop1(x)
        x = self._fc6(x)
        x = self._drop2(x)
        x = self._fc7(x)
        x = self._fc8(x)
        return x

def AlexNet(**args):
    model = AlexNetDY(**args)
    return model
