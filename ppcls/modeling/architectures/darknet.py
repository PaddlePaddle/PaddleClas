# coding=UTF-8
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

__all__ = ["DarkNet53"]


class ConvBNLayer(fluid.dygraph.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 filter_size,
                 stride,
                 padding,
                 name=None):
        super(ConvBNLayer, self).__init__()

        self._conv = Conv2D(
            num_channels=input_channels,
            num_filters=output_channels,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            act=None,
            param_attr=ParamAttr(name=name + ".conv.weights"),
            bias_attr=False)

        bn_name = name + ".bn"
        self._bn = BatchNorm(
            num_channels=output_channels,
            act="relu",
            param_attr=ParamAttr(name=bn_name + ".scale"),
            bias_attr=ParamAttr(name=bn_name + ".offset"),
            moving_mean_name=bn_name + ".mean",
            moving_variance_name=bn_name + ".var")

    def forward(self, inputs):
        x = self._conv(inputs)
        x = self._bn(x)
        return x


class Basic_Block(fluid.dygraph.Layer):
    def __init__(self, input_channels, output_channels, name=None):
        super(Basic_Block, self).__init__()

        self._conv1 = ConvBNLayer(
            input_channels, output_channels, 1, 1, 0, name=name + ".0")
        self._conv2 = ConvBNLayer(
            output_channels, output_channels * 2, 3, 1, 1, name=name + ".1")

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)
        return fluid.layers.elementwise_add(x=inputs, y=x)


class DarkNet(fluid.dygraph.Layer):
    def __init__(self, class_dim=1000):
        super(DarkNet53, self).__init__()

        self.stages = [1, 2, 8, 8, 4]
        self._conv1 = ConvBNLayer(3, 32, 3, 1, 1, name="yolo_input")
        self._conv2 = ConvBNLayer(
            32, 64, 3, 2, 1, name="yolo_input.downsample")

        self._basic_block_01 = Basic_Block(64, 32, name="stage.0.0")
        self._downsample_0 = ConvBNLayer(
            64, 128, 3, 2, 1, name="stage.0.downsample")

        self._basic_block_11 = Basic_Block(128, 64, name="stage.1.0")
        self._basic_block_12 = Basic_Block(128, 64, name="stage.1.1")
        self._downsample_1 = ConvBNLayer(
            128, 256, 3, 2, 1, name="stage.1.downsample")

        self._basic_block_21 = Basic_Block(256, 128, name="stage.2.0")
        self._basic_block_22 = Basic_Block(256, 128, name="stage.2.1")
        self._basic_block_23 = Basic_Block(256, 128, name="stage.2.2")
        self._basic_block_24 = Basic_Block(256, 128, name="stage.2.3")
        self._basic_block_25 = Basic_Block(256, 128, name="stage.2.4")
        self._basic_block_26 = Basic_Block(256, 128, name="stage.2.5")
        self._basic_block_27 = Basic_Block(256, 128, name="stage.2.6")
        self._basic_block_28 = Basic_Block(256, 128, name="stage.2.7")
        self._downsample_2 = ConvBNLayer(
            256, 512, 3, 2, 1, name="stage.2.downsample")

        self._basic_block_31 = Basic_Block(512, 256, name="stage.3.0")
        self._basic_block_32 = Basic_Block(512, 256, name="stage.3.1")
        self._basic_block_33 = Basic_Block(512, 256, name="stage.3.2")
        self._basic_block_34 = Basic_Block(512, 256, name="stage.3.3")
        self._basic_block_35 = Basic_Block(512, 256, name="stage.3.4")
        self._basic_block_36 = Basic_Block(512, 256, name="stage.3.5")
        self._basic_block_37 = Basic_Block(512, 256, name="stage.3.6")
        self._basic_block_38 = Basic_Block(512, 256, name="stage.3.7")
        self._downsample_3 = ConvBNLayer(
            512, 1024, 3, 2, 1, name="stage.3.downsample")

        self._basic_block_41 = Basic_Block(1024, 512, name="stage.4.0")
        self._basic_block_42 = Basic_Block(1024, 512, name="stage.4.1")
        self._basic_block_43 = Basic_Block(1024, 512, name="stage.4.2")
        self._basic_block_44 = Basic_Block(1024, 512, name="stage.4.3")

        self._pool = Pool2D(pool_type="avg", global_pooling=True)

        stdv = 1.0 / math.sqrt(1024.0)
        self._out = Linear(
            input_dim=1024,
            output_dim=class_dim,
            param_attr=ParamAttr(
                name="fc_weights",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"))

    def forward(self, inputs):
        x = self._conv1(inputs)
        x = self._conv2(x)

        x = self._basic_block_01(x)
        x = self._downsample_0(x)

        x = self._basic_block_11(x)
        x = self._basic_block_12(x)
        x = self._downsample_1(x)

        x = self._basic_block_21(x)
        x = self._basic_block_22(x)
        x = self._basic_block_23(x)
        x = self._basic_block_24(x)
        x = self._basic_block_25(x)
        x = self._basic_block_26(x)
        x = self._basic_block_27(x)
        x = self._basic_block_28(x)
        x = self._downsample_2(x)

        x = self._basic_block_31(x)
        x = self._basic_block_32(x)
        x = self._basic_block_33(x)
        x = self._basic_block_34(x)
        x = self._basic_block_35(x)
        x = self._basic_block_36(x)
        x = self._basic_block_37(x)
        x = self._basic_block_38(x)
        x = self._downsample_3(x)

        x = self._basic_block_41(x)
        x = self._basic_block_42(x)
        x = self._basic_block_43(x)
        x = self._basic_block_44(x)

        x = self._pool(x)
        x = fluid.layers.squeeze(x, axes=[2, 3])
        x = self._out(x)
        return x


def DarkNet53(**args):
    model = DarkNet(**args)
    return model
