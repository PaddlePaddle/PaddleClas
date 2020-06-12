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
        x = self.conv_bn_layer(
            input=input,
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
            hidden_channel = int(
                self._make_divisible(exp_size * self.scale, 4))
            x = self.ghost_bottleneck(
                inp=x,
                hidden_dim=hidden_channel,
                oup=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                name="ghost_bottle_" + str(idx))
            idx += 1
        # build last several layers
        output_channel = int(
            self._make_divisible(exp_size * self.scale, 4))
        x = self.conv_bn_layer(
            input=x,
            num_filters=output_channel,
            filter_size=1,
            stride=1,
            groups=1,
            act="relu",
            name="conv2")
        x = fluid.layers.pool2d(
            input=x, pool_type='avg', global_pooling=True)
        output_channel = 1280

        stdv = 1.0 / math.sqrt(x.shape[1] * 1.0)
        out = self.conv_bn_layer(
            input=x,
            num_filters=output_channel,
            filter_size=1,
            stride=1,
            groups=1,
            act="relu",
            name="fc_0")
        out = fluid.layers.dropout(x=out, dropout_prob=0.2)
        stdv = 1.0 / math.sqrt(out.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=out,
            size=class_dim,
            param_attr=ParamAttr(
                name="fc_1_weight",
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
        x = fluid.layers.conv2d(
            input=input,
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=(filter_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(
                initializer=fluid.initializer.MSRA(), name=name + "_weights"),
            bias_attr=False)

        x = fluid.layers.batch_norm(
            input=x,
            act=act,
            param_attr=ParamAttr(
                name=name + "_bn_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=name + "_bn_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance")
        return x

    def se_layer(self, input, num_channels, reduction_ratio=4, name=None):
        pool = fluid.layers.pool2d(
            input=input, pool_size=0, pool_type='avg', global_pooling=True)
        stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
        squeeze = fluid.layers.fc(
            input=pool,
            size=num_channels // reduction_ratio,
            act='relu',
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_sqz_weights'),
            bias_attr=ParamAttr(name=name + '_sqz_offset'))
        stdv = 1.0 / math.sqrt(squeeze.shape[1] * 1.0)
        excitation = fluid.layers.fc(
            input=squeeze,
            size=num_channels,
            act=None,
            param_attr=fluid.param_attr.ParamAttr(
                initializer=fluid.initializer.Uniform(-stdv, stdv),
                name=name + '_exc_weights'),
            bias_attr=ParamAttr(name=name + '_exc_offset'))
        excitation = fluid.layers.clip(
            x=excitation, min=0, max=1)
        se_scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return se_scale

    def depthwise_conv(self,
                       inp,
                       oup,
                       kernel_size,
                       stride=1,
                       relu=False,
                       name=None):
        return self.conv_bn_layer(
            input=inp,
            num_filters=oup,
            filter_size=kernel_size,
            stride=stride,
            groups=inp.shape[1],
            act="relu" if relu else None,
            name=name + "_dw")

    def ghost_module(self,
                    inp,
                    oup,
                    kernel_size=1,
                    ratio=2,
                    dw_size=3,
                    stride=1,
                    relu=True,
                    name=None):
        self.oup = oup
        init_channels = int(math.ceil(oup / ratio))
        new_channels = int(init_channels * (ratio - 1))
        primary_conv = self.conv_bn_layer(
            input=inp,
            num_filters=init_channels,
            filter_size=kernel_size,
            stride=stride,
            groups=1,
            act="relu" if relu else None,
            name=name + "_primary_conv")
        cheap_operation = self.conv_bn_layer(
            input=primary_conv,
            num_filters=new_channels,
            filter_size=dw_size,
            stride=1,
            groups=init_channels,
            act="relu" if relu else None,
            name=name + "_cheap_operation")
        out = fluid.layers.concat(
            [primary_conv, cheap_operation], axis=1)
        return out

    def ghost_bottleneck(self,
                        inp,
                        hidden_dim,
                        oup,
                        kernel_size,
                        stride,
                        use_se,
                        name=None):
        inp_channels = inp.shape[1]
        x = self.ghost_module(
            inp=inp,
            oup=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            name=name + "ghost_module_1")
        if stride == 2:
            x = self.depthwise_conv(
                inp=x,
                oup=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "_dw2")
        if use_se:
            x = self.se_layer(
                input=x, num_channels=hidden_dim, name=name + "se_layer")
        x = self.ghost_module(
            inp=x,
            oup=oup,
            kernel_size=1,
            relu=False,
            name=name + "ghost_module_2")
        if stride == 1 and inp_channels == oup:
            shortcut = inp
        else:
            shortcut = self.depthwise_conv(
                inp=inp,
                oup=inp_channels,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "shortcut_depthwise_conv")
            shortcut = self.conv_bn_layer(
                input=shortcut,
                num_filters=oup,
                filter_size=1,
                stride=1,
                groups=1,
                act=None,
                name=name + "shortcut_conv_bn")
        return fluid.layers.elementwise_add(
            x=x, y=shortcut, axis=-1, act=None)


def GhostNet_x0_5():
    model = GhostNet(scale=0.5)
    return model


def GhostNet_x1_0():
    model = GhostNet(scale=1.0)
    return model


def GhostNet_x1_3():
    model = GhostNet(scale=1.3)
    return model

