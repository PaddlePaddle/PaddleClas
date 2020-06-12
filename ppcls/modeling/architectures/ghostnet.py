from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr

__all__ = ["GhostNet", "GhostNet_0_5", "GhostNet_1_0", "GhostNet_1_3"]


class GhostNet():
    def __init__(self, width_mult):
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
        self.width_mult = width_mult

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
                      name=None,
                      data_format="NCHW"):
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
            bias_attr=False,
            name=name + "_conv_op",
            data_format=data_format)

        x = fluid.layers.batch_norm(
            input=x,
            act=act,
            name=name + "_bn",
            param_attr=ParamAttr(
                name=name + "_bn_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name=name + "_bn_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name=name + "_bn_mean",
            moving_variance_name=name + "_bn_variance",
            data_layout=data_format)
        return x

    def SElayer(self, input, num_channels, reduction_ratio=4, name=None):
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
            x=excitation, min=0, max=1, name=name + '_clip')
        scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
        return scale

    def depthwise_conv(self,
                       inp,
                       oup,
                       kernel_size,
                       stride=1,
                       relu=False,
                       name=None,
                       data_format="NCHW"):
        return self.conv_bn_layer(
            input=inp,
            num_filters=oup,
            filter_size=kernel_size,
            stride=stride,
            groups=inp.shape[1] if data_format == "NCHW" else inp.shape[-1],
            act="relu" if relu else None,
            name=name + "_dw",
            data_format=data_format)

    def GhostModule(self,
                    inp,
                    oup,
                    kernel_size=1,
                    ratio=2,
                    dw_size=3,
                    stride=1,
                    relu=True,
                    name=None,
                    data_format="NCHW"):
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
            name=name + "_primary_conv",
            data_format="NCHW")
        cheap_operation = self.conv_bn_layer(
            input=primary_conv,
            num_filters=new_channels,
            filter_size=dw_size,
            stride=1,
            groups=init_channels,
            act="relu" if relu else None,
            name=name + "_cheap_operation",
            data_format=data_format)
        out = fluid.layers.concat(
            [primary_conv, cheap_operation], axis=1, name=name + "_concat")
        return out

    def GhostBottleneck(self,
                        inp,
                        hidden_dim,
                        oup,
                        kernel_size,
                        stride,
                        use_se,
                        name=None,
                        data_format="NCHW"):
        inp_channels = inp.shape[1]
        x = self.GhostModule(
            inp=inp,
            oup=hidden_dim,
            kernel_size=1,
            stride=1,
            relu=True,
            name=name + "GhostBottle_1",
            data_format="NCHW")
        if stride == 2:
            x = self.depthwise_conv(
                inp=x,
                oup=hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "_dw2",
                data_format="NCHW")
        if use_se:
            x = self.SElayer(
                input=x, num_channels=hidden_dim, name=name + "SElayer")
        x = self.GhostModule(
            inp=x,
            oup=oup,
            kernel_size=1,
            relu=False,
            name=name + "GhostModule_2")
        if stride == 1 and inp_channels == oup:
            shortcut = inp
        else:
            shortcut = self.depthwise_conv(
                inp=inp,
                oup=inp_channels,
                kernel_size=kernel_size,
                stride=stride,
                relu=False,
                name=name + "shortcut_depthwise_conv",
                data_format="NCHW")
            shortcut = self.conv_bn_layer(
                input=shortcut,
                num_filters=oup,
                filter_size=1,
                stride=1,
                groups=1,
                act=None,
                name=name + "shortcut_conv_bn",
                data_format="NCHW")
        return fluid.layers.elementwise_add(
            x=x, y=shortcut, axis=-1, act=None, name=name + "elementwise_add")

    def net(self, input, class_dim=1000):
        # build first layer:
        output_channel = int(self._make_divisible(16 * self.width_mult, 4))
        x = self.conv_bn_layer(
            input=input,
            num_filters=output_channel,
            filter_size=3,
            stride=2,
            groups=1,
            act="relu",
            name="firstlayer",
            data_format="NCHW")
        # build inverted residual blocks
        idx = 0
        for k, exp_size, c, use_se, s in self.cfgs:
            output_channel = int(self._make_divisible(c * self.width_mult, 4))
            hidden_channel = int(
                self._make_divisible(exp_size * self.width_mult, 4))
            x = self.GhostBottleneck(
                inp=x,
                hidden_dim=hidden_channel,
                oup=output_channel,
                kernel_size=k,
                stride=s,
                use_se=use_se,
                name="GhostBottle_" + str(idx),
                data_format="NCHW")
            idx += 1
        # build last several layers
        output_channel = int(
            self._make_divisible(exp_size * self.width_mult, 4))
        x = self.conv_bn_layer(
            input=x,
            num_filters=output_channel,
            filter_size=1,
            stride=1,
            groups=1,
            act="relu",
            name="lastlayer",
            data_format="NCHW")
        x = fluid.layers.pool2d(
            input=x, pool_type='avg', global_pooling=True, data_format="NCHW")
        output_channel = 1280

        stdv = 1.0 / math.sqrt(x.shape[1] * 1.0)
        out = fluid.layers.conv2d(
            input=x,
            num_filters=output_channel,
            filter_size=1,
            groups=1,
            param_attr=ParamAttr(
                name="fc_0_w",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=False,
            name="fc_0")
        out = fluid.layers.batch_norm(
            input=out,
            act="relu",
            name="fc_0_bn",
            param_attr=ParamAttr(
                name="fc_0_bn_scale",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            bias_attr=ParamAttr(
                name="fc_0_bn_offset",
                regularizer=fluid.regularizer.L2DecayRegularizer(
                    regularization_coeff=0.0)),
            moving_mean_name="fc_0_bn_mean",
            moving_variance_name="fc_0_bn_variance",
            data_layout="NCHW")
        out = fluid.layers.dropout(x=out, dropout_prob=0.2)
        stdv = 1.0 / math.sqrt(out.shape[1] * 1.0)
        out = fluid.layers.fc(
            input=out,
            size=class_dim,
            param_attr=ParamAttr(
                name="fc_1_w",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_1_bias"))

        return out


def GhostNet_0_5():
    model = GhostNet(width_mult=0.5)
    return model


def GhostNet_1_0():
    model = GhostNet(width_mult=1.0)
    return model


def GhostNet_1_3():
    model = GhostNet(width_mult=1.3)
    return model
