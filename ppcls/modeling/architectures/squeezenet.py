import numpy as np
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear, Dropout
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid import framework

import math
import sys
import time

__all__ = ["SqueezeNet1_0", "SqueezeNet1_1"]

class Make_Fire_Conv(fluid.dygraph.Layer):
    def __init__(self, 
                input_channels,
                output_channels,
                filter_size,
                padding=0,
                name=None):
        super(Make_Fire_Conv, self).__init__()
        self._conv = Conv2D(input_channels,
                            output_channels,
                            filter_size, 
                            padding=padding, 
                            act="relu",
                            param_attr=ParamAttr(name=name + "_weights"),
                            bias_attr=ParamAttr(name=name + "_offset"))

    def forward(self, inputs):
        return self._conv(inputs)

class Make_Fire(fluid.dygraph.Layer):
    def __init__(self,
                input_channels,
                squeeze_channels,
                expand1x1_channels,
                expand3x3_channels,
                name=None):
        super(Make_Fire, self).__init__()
        self._conv = Make_Fire_Conv(input_channels,
                                    squeeze_channels,
                                    1,
                                    name=name + "_squeeze1x1")
        self._conv_path1 = Make_Fire_Conv(squeeze_channels,
                                        expand1x1_channels,
                                        1,
                                        name=name + "_expand1x1")
        self._conv_path2 = Make_Fire_Conv(squeeze_channels,
                                        expand3x3_channels,
                                        3,
                                        padding=1,
                                        name=name + "_expand3x3")

    def forward(self, inputs):
        x = self._conv(inputs)
        x1 = self._conv_path1(x)
        x2 = self._conv_path2(x)
        return fluid.layers.concat([x1, x2], axis=1)

class SqueezeNet(fluid.dygraph.Layer):
    def __init__(self, version, class_dim=1000):
        super(SqueezeNet, self).__init__()
        self.version = version

        if self.version == "1.0":
            self._conv = Conv2D(3,
                                96,
                                7,
                                stride=2,
                                act="relu",
                                param_attr=ParamAttr(name="conv1_weights"),
                                bias_attr=ParamAttr(name="conv1_offset"))
            self._pool = Pool2D(pool_size=3,
                                pool_stride=2,
                                pool_type="max")
            self._conv1 = Make_Fire(96, 16, 64, 64, name="fire2")
            self._conv2 = Make_Fire(128, 16, 64, 64, name="fire3")
            self._conv3 = Make_Fire(128, 32, 128, 128, name="fire4")

            self._conv4 = Make_Fire(256, 32, 128, 128, name="fire5")
            self._conv5 = Make_Fire(256, 48, 192, 192, name="fire6")
            self._conv6 = Make_Fire(384, 48, 192, 192, name="fire7")
            self._conv7 = Make_Fire(384, 64, 256, 256, name="fire8")

            self._conv8 = Make_Fire(512, 64, 256, 256, name="fire9")
        else:
            self._conv = Conv2D(3,
                                64,
                                3,
                                stride=2,
                                padding=1,
                                act="relu",
                                param_attr=ParamAttr(name="conv1_weights"),
                                bias_attr=ParamAttr(name="conv1_offset"))
            self._pool = Pool2D(pool_size=3,
                                pool_stride=2,
                                pool_type="max")
            self._conv1 = Make_Fire(64, 16, 64, 64, name="fire2")
            self._conv2 = Make_Fire(128, 16, 64, 64, name="fire3")

            self._conv3 = Make_Fire(128, 32, 128, 128, name="fire4")
            self._conv4 = Make_Fire(256, 32, 128, 128, name="fire5")

            self._conv5 = Make_Fire(256, 48, 192, 192, name="fire6")
            self._conv6 = Make_Fire(384, 48, 192, 192, name="fire7")
            self._conv7 = Make_Fire(384, 64, 256, 256, name="fire8")
            self._conv8 = Make_Fire(512, 64, 256, 256, name="fire9")

        self._drop = Dropout(p=0.5)
        self._conv9 = Conv2D(512, 
                            class_dim, 
                            1, 
                            act="relu",
                            param_attr=ParamAttr(name="conv10_weights"),
                            bias_attr=ParamAttr(name="conv10_offset"))
        self._avg_pool = Pool2D(pool_type="avg",
                                global_pooling=True)

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._pool(x)
        if self.version=="1.0":
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._conv3(x)
            x = self._pool(x)
            x = self._conv4(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._pool(x)
            x = self._conv8(x)
        else:
            x = self._conv1(x)
            x = self._conv2(x)
            x = self._pool(x)
            x = self._conv3(x)
            x = self._conv4(x)
            x = self._pool(x)
            x = self._conv5(x)
            x = self._conv6(x)
            x = self._conv7(x)
            x = self._conv8(x)
        x = self._drop(x)
        x = self._conv9(x)
        x = self._avg_pool(x)
        x = fluid.layers.squeeze(x, axes=[2,3])
        return x

def SqueezeNet1_0():
    model = SqueezeNet(version="1.0")
    return model 

def SqueezeNet1_1():
    model = SqueezeNet(version="1.1")
    return model 