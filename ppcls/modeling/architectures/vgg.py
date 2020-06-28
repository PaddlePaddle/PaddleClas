#coding:utf-8
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, BatchNorm, Linear
from paddle.fluid.dygraph.base import to_variable

from paddle.fluid import framework

import math
import sys
import time

__all__ = ["VGG11", "VGG13", "VGG16", "VGG19"]

class Conv_Block(fluid.dygraph.Layer):
    def __init__(self, 
                input_channels, 
                output_channels,
                groups,
                name=None):
        super(Conv_Block, self).__init__()

        self.groups = groups
        self._conv_1 = Conv2D(num_channels=input_channels,
                            num_filters=output_channels,
                            filter_size=3,
                            stride=1,
                            padding=1,
                            act="relu",
                            param_attr=ParamAttr(name=name + "1_weights"),
                            bias_attr=False)
        if groups == 2 or groups == 3 or groups == 4:
            self._conv_2 = Conv2D(num_channels=output_channels,
                                num_filters=output_channels,
                                filter_size=3,
                                stride=1,
                                padding=1,
                                act="relu",
                                param_attr=ParamAttr(name=name + "2_weights"),
                                bias_attr=False)
        if groups == 3 or groups == 4:
            self._conv_3 = Conv2D(num_channels=output_channels,
                                num_filters=output_channels,
                                filter_size=3,
                                stride=1,
                                padding=1,
                                act="relu",
                                param_attr=ParamAttr(name=name + "3_weights"),
                                bias_attr=False)
        if groups == 4:
            self._conv_4 = Conv2D(number_channels=output_channels,
                                number_filters=output_channels,
                                filter_size=3,
                                stride=1,
                                padding=1,
                                act="relu",
                                param_attr=ParamAttr(name=name + "4_weights"),
                                bias_attr=False)
        self._pool = Pool2D(pool_size=2,
                            pool_type="max",
                            pool_stride=2)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        if self.groups == 2 or self.groups == 3 or self.groups == 4:
            x = self._conv_2(x)
        if self.groups == 3 or self.groups == 4 :
            x = self._conv_3(x)
        if self.groups == 4:
            x = self._conv_4(x)
        x = self._pool(x)
        return x

class VGGNet(fluid.dygraph.Layer):
    def __init__(self, layers=11, class_dim=1000):
        super(VGGNet, self).__init__()

        self.layers = layers
        self.vgg_configure = {11: [1, 1, 2, 2, 2],
                            13: [2, 2, 2, 2, 2],
                            16: [2, 2, 3, 3, 3],
                            19: [2, 2, 4, 4, 4]}
        assert self.layers in self.vgg_configure.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_configure.keys(), layers)
        self.groups = self.vgg_configure[self.layers]

        self._conv_block_1 = Conv_Block(3, 64, self.groups[0], name="conv1_")
        self._conv_block_2 = Conv_Block(64, 128, self.groups[1], name="conv2_")
        self._conv_block_3 = Conv_Block(128, 256, self.groups[2], name="conv3_")
        self._conv_block_4 = Conv_Block(256, 512, self.groups[3], name="conv4_")
        self._conv_block_5 = Conv_Block(512, 512, self.groups[4], name="conv5_")

        #self._drop = fluid.dygraph.nn.Dropout(p=0.5)
        self._fc1 = Linear(input_dim=7*7*512,
                        output_dim=4096,
                        act="relu",
                        param_attr=ParamAttr(name="fc6_weights"),
                        bias_attr=ParamAttr(name="fc6_offset"))
        self._fc2 = Linear(input_dim=4096,
                        output_dim=4096,
                        act="relu",
                        param_attr=ParamAttr(name="fc7_weights"),
                        bias_attr=ParamAttr(name="fc7_offset"))
        self._out = Linear(input_dim=4096,
                        output_dim=class_dim,
                        param_attr=ParamAttr(name="fc8_weights"),
                        bias_attr=ParamAttr(name="fc8_offset"))

    def forward(self, inputs):
        x = self._conv_block_1(inputs)
        x = self._conv_block_2(x)
        x = self._conv_block_3(x)
        x = self._conv_block_4(x)
        x = self._conv_block_5(x)

        x = fluid.layers.flatten(x, axis=0)
        x = self._fc1(x)
        # x = self._drop(x)
        x = self._fc2(x)
        # x = self._drop(x)
        x = self._out(x)
        return x

def VGG11():
    model = VGGNet(layers=11)
    return model 

def VGG13():
    model = VGGNet(layers=13)
    return model

def VGG16():
    model = VGGNet(layers=16)
    return model 

def VGG19():
    model = VGGNet(layers=19)
    return model 