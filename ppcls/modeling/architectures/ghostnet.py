#copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ["GhostNet", "GhostNet_x0_5", "GhostNet_x1_0", "GhostNet_x1_3"]


class GhostNet():
    def __init__(self, scale):
        cfgs = [
            # k, t, c, SE, s
            [3, 16, 16, 0, 1],
            [3, 48, 24, 0, 2],
            [3, 72, 24, 0, 1],
            [5, 72, 40, 1, 2],
            [5, 120, 40, 1, 1],
            [3, 240, 80, 0, 2],
            [3, 200, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 184, 80, 0, 1],
            [3, 480, 112, 1, 1],
            [3, 672, 112, 1, 1],
            [5, 672, 160, 1, 2],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1],
            [5, 960, 160, 0, 1],
            [5, 960, 160, 1, 1]
        ]
        self.cfgs = cfgs
        self.scale = scale

    def net(self, input, class_dim=1000):
        # build first layer:
        output_channel = int(self._make_divisible(16 * self.scale, 4))
        x = self.conv_bn_layer(input=input,
                            num_filters=output_channel,
                            filter_size=3,
                            stride=2,
                            groups=1,
                            act="relu",
                            name="conv1")
        # build inverted residual blocks
        idx = 0
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = int(self._make_divisible(c * self.scale, 4))
            hidden_channel = int(self._make_divisible(exp_size * self.scale, 4))
            x = self.ghost_bottleneck(input=x,
                                    hidden_dim=hidden_channel,
                                    output=output_channel,
                                    kernel_size=k,
                                    stride=s,
                                    use_se=use_se,
                                    name="_ghostbottleneck_" + str(idx))
            idx += 1
        # build last several layers
        output_channel = int(self._make_divisible(exp_size * self.scale, 4))
        x = self.conv_bn_layer(input=x,
                            num_filters=output_channel,
                            filter_size=1,
                            stride=1,
                            groups=1,
                            act="relu",
                            name="conv_last")
        x = fluid.layers.pool2d(input=x, pool_type='avg', global_pooling=True)
        output_channel = 1280

        stdv = 1.0 / math.sqrt(x.shape[1] * 1.0)
        out = self.conv_bn_layer(input=x,
                          num_filters=output_channel,
                          filter_size=1,
                          stride=1,
                          act="relu",
                          name="fc_0")
        out = fluid.layers.dropout(x=out, dropout_prob=0.2)
        stdv = 1.0 / math.sqrt(out.shape[1] * 1.0)
        out = fluid.layers.fc(input=out,
                            size=class_dim,
                            param_attr=ParamAttr(name="fc_1_weights",
                                initializer=fluid.initializer.Uniform(-stdv, stdv)),
                            bias_attr=ParamAttr(name="fc_1_offset"))

        return out    
        
    def _make_divisible(self, v, divisor, min_value=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def conv_bn_layer(self,
                      input,
                      num_filters,
                      filter_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        x = fluid.layers.conv2d(input=input,
                                num_filters=num_filters,
                                filter_size=filter_size,
                                stride=stride,
                                padding=(filter_size - 1) // 2,
                                groups=groups,
                                act=None,
                                param_attr=ParamAttr(
                                    initializer=fluid.initializer.MSRA(), name=name + "_weights"),
                                bias_attr=False)
        bn_name = name + "_bn"
        x = fluid.layers.batch_norm(input=x,
                                    act=act,
                                    param_attr=ParamAttr(
                                        name=bn_name + "_scale",
                                        regularizer=fluid.regularizer.L2DecayRegularizer(
                                        regularization_coeff=0.0)),
                                    bias_attr=ParamAttr(
                                        name=bn_name + "_offset",
                                        regularizer=fluid.regularizer.L2DecayRegularizer(
                                        regularization_coeff=0.0)),
                                    moving_mean_name=bn_name + "_mean",
                                    moving_variance_name=name + "_variance")
        return x

    def se_block(self, input, num_channels, reduction_ratio=4, name=None):
        pool = fluid.layers.pool2d(input=input, pool_type='avg', global_pooling=True, use_cudnn=False)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(input=pool,
                                size=num_channels // reduction_ratio,
                                act='relu',
                                param_attr=fluid.param_attr.ParamAttr(
                                    initializer=fluid.initializer.Uniform(-stdv, stdv),
                                    name=name + '_1_weights'),
                                bias_attr=ParamAttr(name=name + '_1_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(input=squeeze,
                                    size=num_channels,
                                    act=None,
                                    param_attr=fluid.param_attr.ParamAttr(
                                        initializer=fluid.initializer.Uniform(-stdv, stdv),
                                        name=name + '_2_weights'),
                                    bias_attr=ParamAttr(name=name + '_2_offset'))
        excitation = fluid.layers.clip(x=excitation, min=0, max=1)
        se_scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return se_scale

    def depthwise_conv(self,
                       input,
                       output,
                       kernel_size,
                       stride=1,
                       relu=False,
                       name=None):
        return self.conv_bn_layer(input=input,
                                num_filters=output,
                                filter_size=kernel_size,
                                stride=stride,
                                groups=input.shape[1],
                                act="relu" if relu else None,
                                name=name + "_depthwise")

    def ghost_module(self,
                    input,
                    output,
                    kernel_size=1,
                    ratio=2,
                    dw_size=3,
                    stride=1,
                    relu=True,
                    name=None):
        self.output = output
        init_channels = int(math.ceil(output / ratio))
        new_channels = int(init_channels * (ratio - 1))
        primary_conv = self.conv_bn_layer(input=input,
                                        num_filters=init_channels,
                                        filter_size=kernel_size,
                                        stride=stride,
                                        groups=1,
                                        act="relu" if relu else None,
                                        name=name + "_primary_conv")
        cheap_operation = self.conv_bn_layer(input=primary_conv,
                                            num_filters=new_channels,
                                            filter_size=dw_size,
                                            stride=1,
                                            groups=init_channels,
                                            act="relu" if relu else None,
                                            name=name + "_cheap_operation")
        out = fluid.layers.concat([primary_conv, cheap_operation], axis=1)
        return out

    def ghost_bottleneck(self,
                        input,
                        hidden_dim,
                        output,
                        kernel_size,
                        stride,
                        use_se,
                        name=None):
        inp_channels = input.shape[1]
        x = self.ghost_module(input=input,
                            output=hidden_dim,
                            kernel_size=1,
                            stride=1,
                            relu=True,
                            name=name + "_ghost_module_1")
        if stride == 2:
            x = self.depthwise_conv(input=x,
                                    output=hidden_dim,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    relu=False,
                                    name=name + "_depthwise")
        if use_se:
            x = self.se_block(input=x, num_channels=hidden_dim, name=name + "_se")
        x = self.ghost_module(input=x,
                            output=output,
                            kernel_size=1,
                            relu=False,
                            name=name + "_ghost_module_2")
        if stride == 1 and inp_channels == output:
            shortcut = input
        else:
            shortcut = self.depthwise_conv(input=input,
                                        output=inp_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        relu=False,
                                        name=name + "_shortcut_depthwise")
            shortcut = self.conv_bn_layer(input=shortcut,
                                        num_filters=output,
                                        filter_size=1,
                                        stride=1,
                                        groups=1,
                                        act=None,
                                        name=name + "_shortcut_conv")
        return fluid.layers.elementwise_add(x=x, 
                                            y=shortcut,
                                            axis=-1)


def GhostNet_x0_5():
    model = GhostNet(scale=0.5)
    return model


def GhostNet_x1_0():
    model = GhostNet(scale=1.0)
    return model


def GhostNet_x1_3():
    model = GhostNet(scale=1.3)
    return model
