from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid as fluid
from paddle.fluid.initializer import MSRA, ConstantInitializer
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.regularizer import L2DecayRegularizer
import math

__all__ = [
    'ResNeSt50', 'ResNeSt101', 'ResNeSt200', 'ResNeSt269',
    'ResNeSt50_fast_1s1x64d', 'ResNeSt50_fast_2s1x64d',
    'ResNeSt50_fast_4s1x64d', 'ResNeSt50_fast_1s2x40d',
    'ResNeSt50_fast_2s2x40d', 'ResNeSt50_fast_2s2x40d',
    'ResNeSt50_fast_4s2x40d', 'ResNeSt50_fast_1s4x24d'
]


class ResNeSt():
    def __init__(self,
                 layers,
                 radix=1,
                 groups=1,
                 bottleneck_width=64,
                 dilated=False,
                 dilation=1,
                 deep_stem=False,
                 stem_width=64,
                 avg_down=False,
                 rectify_avg=False,
                 avd=False,
                 avd_first=False,
                 final_drop=0.0,
                 last_gamma=False,
                 bn_decay=0.0):
        self.cardinality = groups
        self.bottleneck_width = bottleneck_width
        # ResNet-D params
        self.inplanes = stem_width * 2 if deep_stem else 64
        self.avg_down = avg_down
        self.last_gamma = last_gamma
        # ResNeSt params
        self.radix = radix
        self.avd = avd
        self.avd_first = avd_first

        self.deep_stem = deep_stem
        self.stem_width = stem_width
        self.layers = layers
        self.final_drop = final_drop
        self.dilated = dilated
        self.dilation = dilation
        self.bn_decay = bn_decay

        self.rectify_avg = rectify_avg

    def net(self, input, class_dim=1000):
        if self.deep_stem:
            x = self.conv_bn_layer(
                x=input,
                num_filters=self.stem_width,
                filters_size=3,
                stride=2,
                groups=1,
                act="relu",
                name="conv1")
            x = self.conv_bn_layer(
                x=x,
                num_filters=self.stem_width,
                filters_size=3,
                stride=1,
                groups=1,
                act="relu",
                name="conv2")
            x = self.conv_bn_layer(
                x=x,
                num_filters=self.stem_width * 2,
                filters_size=3,
                stride=1,
                groups=1,
                act="relu",
                name="conv3")
        else:
            x = self.conv_bn_layer(
                x=input,
                num_filters=64,
                filters_size=7,
                stride=2,
                act="relu",
                name="conv1")

        x = fluid.layers.pool2d(
            input=x,
            pool_size=3,
            pool_type="max",
            pool_stride=2,
            pool_padding=1)

        x = self.resnest_layer(
            x=x,
            planes=64,
            blocks=self.layers[0],
            is_first=False,
            name="layer1")
        x = self.resnest_layer(
            x=x, 
            planes=128, 
            blocks=self.layers[1], 
            stride=2, 
            name="layer2")
        if self.dilated or self.dilation == 4:
            x = self.resnest_layer(
                x=x,
                planes=256,
                blocks=self.layers[2],
                stride=1,
                dilation=2,
                name="layer3")
            x = self.resnest_layer(
                x=x,
                planes=512,
                blocks=self.layers[3],
                stride=1,
                dilation=4,
                name="layer4")
        elif self.dilation == 2:
            x = self.resnest_layer(
                x=x,
                planes=256,
                blocks=self.layers[2],
                stride=2,
                dilation=1,
                name="layer3")
            x = self.resnest_layer(
                x=x,
                planes=512,
                blocks=self.layers[3],
                stride=1,
                dilation=2,
                name="layer4")
        else:
            x = self.resnest_layer(
                x=x,
                planes=256,
                blocks=self.layers[2],
                stride=2,
                name="layer3")
            x = self.resnest_layer(
                x=x,
                planes=512,
                blocks=self.layers[3],
                stride=2,
                name="layer4")
        x = fluid.layers.pool2d(
            input=x, pool_type="avg", global_pooling=True)
        x = fluid.layers.dropout(
            x=x, dropout_prob=self.final_drop)
        stdv = 1.0 / math.sqrt(x.shape[1] * 1.0)
        x = fluid.layers.fc(
            input=x,
            size=class_dim,
            param_attr=ParamAttr(
                name="fc_weights",
                initializer=fluid.initializer.Uniform(-stdv, stdv)),
            bias_attr=ParamAttr(name="fc_offset"))
        return x

    def conv_bn_layer(self,
                      x,
                      num_filters,
                      filters_size,
                      stride=1,
                      groups=1,
                      act=None,
                      name=None):
        x = fluid.layers.conv2d(
            input=x,
            num_filters=num_filters,
            filter_size=filters_size,
            stride=stride,
            padding=(filters_size - 1) // 2,
            groups=groups,
            act=None,
            param_attr=ParamAttr(
                initializer=MSRA(), name=name + "_weight"),
            bias_attr=False)
        x = fluid.layers.batch_norm(
            input=x,
            act=act,
            param_attr=ParamAttr(
                name=name + "_scale",
                regularizer=L2DecayRegularizer(
                    regularization_coeff=self.bn_decay)),
            bias_attr=ParamAttr(
                name=name + "_offset",
                regularizer=L2DecayRegularizer(
                    regularization_coeff=self.bn_decay)),
            moving_mean_name=name + "_mean",
            moving_variance_name=name + "_variance")
        return x

    def rsoftmax(self, x, radix, cardinality):
        batch, r, h, w = x.shape
        if radix > 1:
            x = fluid.layers.reshape(
                x=x,
                shape=[
                    0, cardinality, radix, int(r * h * w / cardinality / radix)
                ])
            x = fluid.layers.transpose(x=x, perm=[0, 2, 1, 3])
            x = fluid.layers.softmax(input=x, axis=1)
            x = fluid.layers.reshape(x=x, shape=[0, r * h * w])
        else:
            x = fluid.layers.sigmoid(x=x)
        return x

    def splat_conv(self,
                   x,
                   in_channels,
                   channels,
                   kernel_size,
                   stride=1,
                   padding=0,
                   dilation=1,
                   groups=1,
                   bias=True,
                   radix=2,
                   reduction_factor=4,
                   rectify_avg=False,
                   name=None):
        x = self.conv_bn_layer(
            x=x,
            num_filters=channels * radix,
            filters_size=kernel_size,
            stride=stride,
            groups=groups * radix,
            act="relu",
            name=name + "_splat1")

        batch, rchannel = x.shape[:2]
        if radix > 1:
            splited = fluid.layers.split(input=x, num_or_sections=radix, dim=1)
            gap = fluid.layers.sum(x=splited)
        else:
            gap = x
        gap = fluid.layers.pool2d(
            input=gap, pool_type="avg", global_pooling=True)
        inter_channels = int(max(in_channels * radix // reduction_factor, 32))
        gap = self.conv_bn_layer(
            x=gap,
            num_filters=inter_channels,
            filters_size=1,
            groups=groups,
            act="relu",
            name=name + "_splat2")

        atten = fluid.layers.conv2d(
            input=gap,
            num_filters=channels * radix,
            filter_size=1,
            stride=1,
            padding=0,
            groups=groups,
            act=None,
            param_attr=ParamAttr(
                name=name + "_splat_weights", initializer=MSRA()),
            bias_attr=False)
        atten = self.rsoftmax(
            x=atten, radix=radix, cardinality=groups)
        atten = fluid.layers.reshape(x=atten, shape=[-1, atten.shape[1], 1, 1])

        if radix > 1:
            attens = fluid.layers.split(
                input=atten, num_or_sections=radix, dim=1)
            out = fluid.layers.sum([
                fluid.layers.elementwise_mul(
                    x=att, y=split) for (att, split) in zip(attens, splited)
            ])
        else:
            out = fluid.layers.elementwise_mul(atten, x)
        return out

    def bottleneck(self,
                   x,
                   inplanes,
                   planes,
                   stride=1,
                   radix=1,
                   cardinality=1,
                   bottleneck_width=64,
                   avd=False,
                   avd_first=False,
                   dilation=1,
                   is_first=False,
                   rectify_avg=False,
                   last_gamma=False,
                   name=None):

        short = x

        group_width = int(planes * (bottleneck_width / 64.)) * cardinality
        x = self.conv_bn_layer(
            x=x,
            num_filters=group_width,
            filters_size=1,
            stride=1,
            groups=1,
            act="relu",
            name=name + "_conv1")
        if avd and avd_first and (stride > 1 or is_first):
            x = fluid.layers.pool2d(
                input=x,
                pool_size=3,
                pool_type="avg",
                pool_stride=stride,
                pool_padding=1)
        if radix >= 1:
            x = self.splat_conv(
                x=x,
                in_channels=group_width,
                channels=group_width,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation,
                groups=cardinality,
                bias=False,
                radix=radix,
                rectify_avg=rectify_avg,
                name=name + "_splatconv")
        else:
            x = self.conv_bn_layer(
                x=x,
                num_filters=group_width,
                filters_size=3,
                stride=1,
                padding=dilation,
                dilation=dialtion,
                groups=cardinality,
                act="relu",
                name=name + "_conv2")

        if avd and avd_first == False and (stride > 1 or is_first):
            x = fluid.layers.pool2d(
                input=x,
                pool_size=3,
                pool_type="avg",
                pool_stride=stride,
                pool_padding=1)
        x = self.conv_bn_layer(
            x=x,
            num_filters=planes * 4,
            filters_size=1,
            stride=1,
            groups=1,
            act=None,
            name=name + "_conv3")

        if stride != 1 or self.inplanes != planes * 4:
            if self.avg_down:
                if dilation == 1:
                    short = fluid.layers.pool2d(
                        input=short,
                        pool_size=stride,
                        pool_type="avg",
                        pool_stride=stride,
                        ceil_mode=True)
                else:
                    short = fluid.layers.pool2d(
                        input=short,
                        pool_size=1,
                        pool_type="avg",
                        pool_stride=1,
                        ceil_mode=True)
                short = fluid.layers.conv2d(
                    input=short,
                    num_filters=planes * 4,
                    filter_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    act=None,
                    param_attr=ParamAttr(
                        name=name + "_weights", initializer=MSRA()),
                    bias_attr=False)
            else:
                short = fluid.layers.conv2d(
                    input=short,
                    num_filters=planes * 4,
                    filter_size=1,
                    stride=stride,
                    param_attr=ParamAttr(
                        name=name + "_shortcut_weights", initializer=MSRA()),
                    bias_attr=False)

            short = fluid.layers.batch_norm(
                input=short,
                act=None,
                param_attr=ParamAttr(
                    name=name + "_shortcut_scale",
                    regularizer=L2DecayRegularizer(
                        regularization_coeff=self.bn_decay)),
                bias_attr=ParamAttr(
                    name=name + "_shortcut_offset",
                    regularizer=L2DecayRegularizer(
                        regularization_coeff=self.bn_decay)),
                moving_mean_name=name + "_shortcut_mean",
                moving_variance_name=name + "_shortcut_variance")

        return fluid.layers.elementwise_add(x=short, y=x, act="relu")

    def resnest_layer(self,
                      x,
                      planes,
                      blocks,
                      stride=1,
                      dilation=1,
                      is_first=True,
                      name=None):
        if dilation == 1 or dilation == 2:
            x = self.bottleneck(
                x=x,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                radix=self.radix,
                cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width,
                avd=self.avd,
                avd_first=self.avd_first,
                dilation=1,
                is_first=is_first,
                rectify_avg=self.rectify_avg,
                last_gamma=self.last_gamma,
                name=name + "_bottleneck_0")
        elif dilation == 4:
            x = self.bottleneck(
                x=x,
                inplanes=self.inplanes,
                planes=planes,
                stride=stride,
                radix=self.radix,
                cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width,
                avd=self.avd,
                avd_first=self.avd_first,
                dilation=2,
                is_first=is_first,
                rectify_avg=self.rectify_avg,
                last_gamma=self.last_gamma,
                name=name + "_bottleneck_0")
        else:
            raise RuntimeError("=>unknown dilation size")

        self.inplanes = planes * 4
        for i in range(1, blocks):
            name = name + "_bottleneck_" + str(i)
            x = self.bottleneck(
                x=x,
                inplanes=self.inplanes,
                planes=planes,
                radix=self.radix,
                cardinality=self.cardinality,
                bottleneck_width=self.bottleneck_width,
                avd=self.avd,
                avd_first=self.avd_first,
                dilation=dilation,
                rectify_avg=self.rectify_avg,
                last_gamma=self.last_gamma,
                name=name)
        return x


def ResNeSt50(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **args)
    return model


def ResNeSt101(**args):
    model = ResNeSt(
        layers=[3, 4, 23, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.0,
        **args)
    return model


def ResNeSt200(**args):
    model = ResNeSt(
        layers=[3, 24, 36, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.2,
        **args)
    return model


def ResNeSt269(**args):
    model = ResNeSt(
        layers=[3, 30, 48, 8],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=64,
        avg_down=True,
        avd=True,
        avd_first=False,
        final_drop=0.2,
        **args)
    return model


def ResNeSt50_fast_1s1x64d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=1,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_2s1x64d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_4s1x64d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=1,
        bottleneck_width=64,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_1s2x40d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=1,
        groups=2,
        bottleneck_width=40,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_2s2x40d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=2,
        groups=2,
        bottleneck_width=40,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_4s2x40d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=4,
        groups=2,
        bottleneck_width=40,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model


def ResNeSt50_fast_1s4x24d(**args):
    model = ResNeSt(
        layers=[3, 4, 6, 3],
        radix=1,
        groups=4,
        bottleneck_width=24,
        deep_stem=True,
        stem_width=32,
        avg_down=True,
        avd=True,
        avd_first=True,
        final_drop=0.0,
        **args)
    return model
