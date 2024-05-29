# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code was based on https://github.com/lukemelas/EfficientNet-PyTorch
# reference: https://arxiv.org/abs/1905.11946

import math
import re

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal, Uniform
from paddle.regularizer import L2Decay

from ....utils.config import AttrDict

from ....utils.save_load import (load_dygraph_pretrain,
                                 load_dygraph_pretrain)

MODEL_URLS = {
    "EfficientNetV2_S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetV2_S_pretrained.pdparams",
    "EfficientNetV2_M":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetV2_M_pretrained.pdparams",
    "EfficientNetV2_L":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetV2_L_pretrained.pdparams",
    "EfficientNetV2_XL":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientNetV2_XL_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

inp_shape = {
    "efficientnetv2-s": [384, 192, 192, 96, 48, 24, 24, 12],
    "efficientnetv2-m": [384, 192, 192, 96, 48, 24, 24, 12],
    "efficientnetv2-l": [384, 192, 192, 96, 48, 24, 24, 12],
    "efficientnetv2-xl": [384, 192, 192, 96, 48, 24, 24, 12],
}


def cal_padding(img_size, stride, kernel_size):
    """Calculate padding size."""
    if img_size % stride == 0:
        out_size = max(kernel_size - stride, 0)
    else:
        out_size = max(kernel_size - (img_size % stride), 0)
    return out_size // 2, out_size - out_size // 2


class Conv2ds(nn.Layer):
    """Customed Conv2D with tensorflow's padding style

    Args:
        input_channels (int): input channels
        output_channels (int): output channels
        kernel_size (int): filter size
        stride (int, optional): stride. Defaults to 1.
        padding (int, optional): padding. Defaults to 0.
        groups (int, optional): groups. Defaults to None.
        act (str, optional): act. Defaults to None.
        use_bias (bool, optional): use_bias. Defaults to None.
        padding_type (str, optional): padding_type. Defaults to None.
        model_name (str, optional): model name. Defaults to None.
        cur_stage (int, optional): current stage. Defaults to None.

    Returns:
        nn.Layer: Customed Conv2D instance
    """

    def __init__(self,
                 input_channels: int,
                 output_channels: int,
                 kernel_size: int,
                 stride=1,
                 padding=0,
                 groups=None,
                 act=None,
                 use_bias=None,
                 padding_type=None,
                 model_name=None,
                 cur_stage=None):
        super(Conv2ds, self).__init__()
        assert act in [None, "swish", "sigmoid"]
        self._act = act

        def get_padding(kernel_size, stride=1, dilation=1):
            padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
            return padding

        inps = inp_shape[model_name][cur_stage]
        self.need_crop = False
        if padding_type == "SAME":
            top_padding, bottom_padding = cal_padding(inps, stride,
                                                      kernel_size)
            left_padding, right_padding = cal_padding(inps, stride,
                                                      kernel_size)
            height_padding = bottom_padding
            width_padding = right_padding
            if top_padding != bottom_padding or left_padding != right_padding:
                height_padding = top_padding + stride
                width_padding = left_padding + stride
                self.need_crop = True
            padding = [height_padding, width_padding]
        elif padding_type == "VALID":
            height_padding = 0
            width_padding = 0
            padding = [height_padding, width_padding]
        elif padding_type == "DYNAMIC":
            padding = get_padding(kernel_size, stride)
        else:
            padding = padding_type

        groups = 1 if groups is None else groups
        self._conv = nn.Conv2D(
            input_channels,
            output_channels,
            kernel_size,
            groups=groups,
            stride=stride,
            padding=padding,
            weight_attr=None,
            bias_attr=use_bias
            if not use_bias else ParamAttr(regularizer=L2Decay(0.0)))

    def forward(self, inputs):
        x = self._conv(inputs)
        if self._act == "swish":
            x = F.swish(x)
        elif self._act == "sigmoid":
            x = F.sigmoid(x)

        if self.need_crop:
            x = x[:, :, 1:, 1:]
        return x


class BlockDecoder(object):
    """Block Decoder for readability."""

    def _decode_block_string(self, block_string):
        """Gets a block through a string notation of arguments."""
        assert isinstance(block_string, str)
        ops = block_string.split('_')
        options = AttrDict()
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        t = AttrDict(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            in_channels=int(options['i']),
            out_channels=int(options['o']),
            expand_ratio=int(options['e']),
            se_ratio=float(options['se']) if 'se' in options else None,
            strides=int(options['s']),
            conv_type=int(options['c']) if 'c' in options else 0, )
        return t

    def _encode_block_string(self, block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d' % block.strides,
            'e%s' % block.expand_ratio,
            'i%d' % block.in_channels,
            'o%d' % block.out_channels,
            'c%d' % block.conv_type,
            'f%d' % block.fused_conv,
        ]
        if block.se_ratio > 0 and block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        return '_'.join(args)

    def decode(self, string_list):
        """Decodes a list of string notations to specify blocks inside the network.

        Args:
        string_list: a list of strings, each string is a notation of block.

        Returns:
        A list of namedtuples to represent blocks arguments.
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(self._decode_block_string(block_string))
        return blocks_args

    def encode(self, blocks_args):
        """Encodes a list of Blocks to a list of strings.

        Args:
        blocks_args: A list of namedtuples to represent blocks arguments.
        Returns:
        a list of strings, each string is a notation of block.
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(self._encode_block_string(block))
        return block_strings


#################### EfficientNet V2 configs ####################
v2_base_block = [  # The baseline config for v2 models.
    "r1_k3_s1_e1_i32_o16_c1",
    "r2_k3_s2_e4_i16_o32_c1",
    "r2_k3_s2_e4_i32_o48_c1",
    "r3_k3_s2_e4_i48_o96_se0.25",
    "r5_k3_s1_e6_i96_o112_se0.25",
    "r8_k3_s2_e6_i112_o192_se0.25",
]

v2_s_block = [  # about base * (width1.4, depth1.8)
    "r2_k3_s1_e1_i24_o24_c1",
    "r4_k3_s2_e4_i24_o48_c1",
    "r4_k3_s2_e4_i48_o64_c1",
    "r6_k3_s2_e4_i64_o128_se0.25",
    "r9_k3_s1_e6_i128_o160_se0.25",
    "r15_k3_s2_e6_i160_o256_se0.25",
]

v2_m_block = [  # about base * (width1.6, depth2.2)
    "r3_k3_s1_e1_i24_o24_c1",
    "r5_k3_s2_e4_i24_o48_c1",
    "r5_k3_s2_e4_i48_o80_c1",
    "r7_k3_s2_e4_i80_o160_se0.25",
    "r14_k3_s1_e6_i160_o176_se0.25",
    "r18_k3_s2_e6_i176_o304_se0.25",
    "r5_k3_s1_e6_i304_o512_se0.25",
]

v2_l_block = [  # about base * (width2.0, depth3.1)
    "r4_k3_s1_e1_i32_o32_c1",
    "r7_k3_s2_e4_i32_o64_c1",
    "r7_k3_s2_e4_i64_o96_c1",
    "r10_k3_s2_e4_i96_o192_se0.25",
    "r19_k3_s1_e6_i192_o224_se0.25",
    "r25_k3_s2_e6_i224_o384_se0.25",
    "r7_k3_s1_e6_i384_o640_se0.25",
]

v2_xl_block = [  # only for 21k pretraining.
    "r4_k3_s1_e1_i32_o32_c1",
    "r8_k3_s2_e4_i32_o64_c1",
    "r8_k3_s2_e4_i64_o96_c1",
    "r16_k3_s2_e4_i96_o192_se0.25",
    "r24_k3_s1_e6_i192_o256_se0.25",
    "r32_k3_s2_e6_i256_o512_se0.25",
    "r8_k3_s1_e6_i512_o640_se0.25",
]
efficientnetv2_params = {
    # params:            (block, width, depth, dropout)
    "efficientnetv2-s":
    (v2_s_block, 1.0, 1.0, np.linspace(0.1, 0.3, 4).tolist()),
    "efficientnetv2-m": (v2_m_block, 1.0, 1.0, 0.3),
    "efficientnetv2-l": (v2_l_block, 1.0, 1.0, 0.4),
    "efficientnetv2-xl": (v2_xl_block, 1.0, 1.0, 0.4),
}


def efficientnetv2_config(model_name: str):
    """EfficientNetV2 model config."""
    block, width, depth, dropout = efficientnetv2_params[model_name]

    cfg = AttrDict(model=AttrDict(
        model_name=model_name,
        blocks_args=BlockDecoder().decode(block),
        width_coefficient=width,
        depth_coefficient=depth,
        dropout_rate=dropout,
        feature_size=1280,
        bn_momentum=0.9,
        bn_epsilon=1e-3,
        depth_divisor=8,
        min_depth=8,
        act_fn="silu",
        survival_prob=0.8,
        local_pooling=False,
        conv_dropout=0,
        num_classes=1000))
    return cfg


def get_model_config(model_name: str):
    """Main entry for model name to config."""
    if model_name.startswith("efficientnetv2-"):
        return efficientnetv2_config(model_name)
    raise ValueError(f"Unknown model_name {model_name}")


################################################################################


def round_filters(filters,
                  width_coefficient,
                  depth_divisor,
                  min_depth,
                  skip=False):
    """Round number of filters based on depth multiplier."""
    multiplier = width_coefficient
    divisor = depth_divisor
    min_depth = min_depth
    if skip or not multiplier:
        return filters

    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth,
                      int(filters + divisor / 2) // divisor * divisor)
    return int(new_filters)


def round_repeats(repeats, multiplier, skip=False):
    """Round number of filters based on depth multiplier."""
    if skip or not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def activation_fn(act_fn: str):
    """Customized non-linear activation type."""
    if not act_fn:
        return nn.Silu()
    elif act_fn in ("silu", "swish"):
        return nn.Swish()
    elif act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "relu6":
        return nn.ReLU6()
    elif act_fn == "elu":
        return nn.ELU()
    elif act_fn == "leaky_relu":
        return nn.LeakyReLU()
    elif act_fn == "selu":
        return nn.SELU()
    elif act_fn == "mish":
        return nn.Mish()
    else:
        raise ValueError("Unsupported act_fn {}".format(act_fn))


def drop_path(x, training=False, survival_prob=1.0):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if not training:
        return x
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    keep_prob = paddle.to_tensor(survival_prob, dtype=x.dtype)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class SE(nn.Layer):
    """Squeeze-and-excitation layer.

    Args:
        local_pooling (bool): local_pooling
        act_fn (str): act_fn
        in_channels (int): in_channels
        se_channels (int): se_channels
        out_channels (int): out_channels
        cur_stage (int): cur_stage
        padding_type (str): padding_type
        model_name (str): model_name
    """

    def __init__(self,
                 local_pooling: bool,
                 act_fn: str,
                 in_channels: int,
                 se_channels: int,
                 out_channels: int,
                 cur_stage: int,
                 padding_type: str,
                 model_name: str):
        super(SE, self).__init__()

        self._local_pooling = local_pooling
        self._act = activation_fn(act_fn)

        # Squeeze and Excitation layer.
        self._se_reduce = Conv2ds(
            in_channels,
            se_channels,
            1,
            stride=1,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)
        self._se_expand = Conv2ds(
            se_channels,
            out_channels,
            1,
            stride=1,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)

    def forward(self, x):
        if self._local_pooling:
            se_tensor = F.adaptive_avg_pool2d(x, output_size=1)
        else:
            se_tensor = paddle.mean(x, axis=[2, 3], keepdim=True)
        se_tensor = self._se_expand(self._act(self._se_reduce(se_tensor)))
        return F.sigmoid(se_tensor) * x


class MBConvBlock(nn.Layer):
    """A class of MBConv: Mobile Inverted Residual Bottleneck.

    Args:
        se_ratio (int): se_ratio
        in_channels (int): in_channels
        expand_ratio (int): expand_ratio
        kernel_size (int): kernel_size
        strides (int): strides
        out_channels (int): out_channels
        bn_momentum (float): bn_momentum
        bn_epsilon (float): bn_epsilon
        local_pooling (bool): local_pooling
        conv_dropout (float): conv_dropout
        cur_stage (int): cur_stage
        padding_type (str): padding_type
        model_name (str): model_name
    """

    def __init__(self,
                 se_ratio: int,
                 in_channels: int,
                 expand_ratio: int,
                 kernel_size: int,
                 strides: int,
                 out_channels: int,
                 bn_momentum: float,
                 bn_epsilon: float,
                 local_pooling: bool,
                 conv_dropout: float,
                 cur_stage: int,
                 padding_type: str,
                 model_name: str):
        super(MBConvBlock, self).__init__()

        self.se_ratio = se_ratio
        self.in_channels = in_channels
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_channels = out_channels

        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self._local_pooling = local_pooling
        self.act_fn = None
        self.conv_dropout = conv_dropout

        self._act = activation_fn(None)
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)
        """Builds block according to the arguments."""
        expand_channels = self.in_channels * self.expand_ratio
        kernel_size = self.kernel_size

        # Expansion phase. Called if not using fused convolutions and expansion
        # phase is necessary.
        if self.expand_ratio != 1:
            self._expand_conv = Conv2ds(
                self.in_channels,
                expand_channels,
                1,
                stride=1,
                use_bias=False,
                padding_type=padding_type,
                model_name=model_name,
                cur_stage=cur_stage)
            self._norm0 = nn.BatchNorm2D(
                expand_channels,
                self.bn_momentum,
                self.bn_epsilon,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        # Depth-wise convolution phase. Called if not using fused convolutions.
        self._depthwise_conv = Conv2ds(
            expand_channels,
            expand_channels,
            kernel_size,
            padding=kernel_size // 2,
            stride=self.strides,
            groups=expand_channels,
            use_bias=False,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)

        self._norm1 = nn.BatchNorm2D(
            expand_channels,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        if self._has_se:
            num_reduced_filters = max(1, int(self.in_channels * self.se_ratio))
            self._se = SE(self._local_pooling, None, expand_channels,
                          num_reduced_filters, expand_channels, cur_stage,
                          padding_type, model_name)
        else:
            self._se = None

        # Output phase.
        self._project_conv = Conv2ds(
            expand_channels,
            self.out_channels,
            1,
            stride=1,
            use_bias=False,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)
        self._norm2 = nn.BatchNorm2D(
            self.out_channels,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.drop_out = nn.Dropout(self.conv_dropout)

    def residual(self, inputs, x, survival_prob):
        if (self.strides == 1 and self.in_channels == self.out_channels):
            # Apply only if skip connection presents.
            if survival_prob:
                x = drop_path(x, self.training, survival_prob)
            x = paddle.add(x, inputs)

        return x

    def forward(self, inputs, survival_prob=None):
        """Implementation of call().

        Args:
            inputs: the inputs tensor.
            survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
            A output tensor.
        """
        x = inputs
        if self.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x)))

        x = self._act(self._norm1(self._depthwise_conv(x)))

        if self.conv_dropout and self.expand_ratio > 1:
            x = self.drop_out(x)

        if self._se:
            x = self._se(x)

        x = self._norm2(self._project_conv(x))
        x = self.residual(inputs, x, survival_prob)

        return x


class FusedMBConvBlock(MBConvBlock):
    """Fusing the proj conv1x1 and depthwise_conv into a conv2d."""

    def __init__(self, se_ratio, in_channels, expand_ratio, kernel_size,
                 strides, out_channels, bn_momentum, bn_epsilon, local_pooling,
                 conv_dropout, cur_stage, padding_type, model_name):
        """Builds block according to the arguments."""
        super(MBConvBlock, self).__init__()
        self.se_ratio = se_ratio
        self.in_channels = in_channels
        self.expand_ratio = expand_ratio
        self.kernel_size = kernel_size
        self.strides = strides
        self.out_channels = out_channels

        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon

        self._local_pooling = local_pooling
        self.act_fn = None
        self.conv_dropout = conv_dropout

        self._act = activation_fn(None)
        self._has_se = (self.se_ratio is not None and 0 < self.se_ratio <= 1)

        expand_channels = self.in_channels * self.expand_ratio
        kernel_size = self.kernel_size
        if self.expand_ratio != 1:
            # Expansion phase:
            self._expand_conv = Conv2ds(
                self.in_channels,
                expand_channels,
                kernel_size,
                padding=kernel_size // 2,
                stride=self.strides,
                use_bias=False,
                padding_type=padding_type,
                model_name=model_name,
                cur_stage=cur_stage)
            self._norm0 = nn.BatchNorm2D(
                expand_channels,
                self.bn_momentum,
                self.bn_epsilon,
                weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))

        if self._has_se:
            num_reduced_filters = max(1, int(self.in_channels * self.se_ratio))
            self._se = SE(self._local_pooling, None, expand_channels,
                          num_reduced_filters, expand_channels, cur_stage,
                          padding_type, model_name)
        else:
            self._se = None

        # Output phase:
        self._project_conv = Conv2ds(
            expand_channels,
            self.out_channels,
            1 if (self.expand_ratio != 1) else kernel_size,
            padding=(1 if (self.expand_ratio != 1) else kernel_size) // 2,
            stride=1 if (self.expand_ratio != 1) else self.strides,
            use_bias=False,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)
        self._norm1 = nn.BatchNorm2D(
            self.out_channels,
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.drop_out = nn.Dropout(conv_dropout)

    def forward(self, inputs, survival_prob=None):
        """Implementation of call().

        Args:
            inputs: the inputs tensor.
            training: boolean, whether the model is constructed for training.
            survival_prob: float, between 0 to 1, drop connect rate.

        Returns:
            A output tensor.
        """
        x = inputs
        if self.expand_ratio != 1:
            x = self._act(self._norm0(self._expand_conv(x)))

        if self.conv_dropout and self.expand_ratio > 1:
            x = self.drop_out(x)

        if self._se:
            x = self._se(x)

        x = self._norm1(self._project_conv(x))
        if self.expand_ratio == 1:
            x = self._act(x)  # add act if no expansion.

        x = self.residual(inputs, x, survival_prob)
        return x


class Stem(nn.Layer):
    """Stem layer at the begining of the network."""

    def __init__(self, width_coefficient, depth_divisor, min_depth, skip,
                 bn_momentum, bn_epsilon, act_fn, stem_channels, cur_stage,
                 padding_type, model_name):
        super(Stem, self).__init__()
        self._conv_stem = Conv2ds(
            3,
            round_filters(stem_channels, width_coefficient, depth_divisor,
                          min_depth, skip),
            3,
            padding=1,
            stride=2,
            use_bias=False,
            padding_type=padding_type,
            model_name=model_name,
            cur_stage=cur_stage)
        self._norm = nn.BatchNorm2D(
            round_filters(stem_channels, width_coefficient, depth_divisor,
                          min_depth, skip),
            bn_momentum,
            bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._act = activation_fn(act_fn)

    def forward(self, inputs):
        return self._act(self._norm(self._conv_stem(inputs)))


class Head(nn.Layer):
    """Head layer for network outputs."""

    def __init__(self,
                 in_channels,
                 feature_size,
                 bn_momentum,
                 bn_epsilon,
                 act_fn,
                 dropout_rate,
                 local_pooling,
                 width_coefficient,
                 depth_divisor,
                 min_depth,
                 skip=False):
        super(Head, self).__init__()
        self.in_channels = in_channels
        self.feature_size = feature_size
        self.bn_momentum = bn_momentum
        self.bn_epsilon = bn_epsilon
        self.dropout_rate = dropout_rate
        self._local_pooling = local_pooling
        self._conv_head = nn.Conv2D(
            in_channels,
            round_filters(self.feature_size or 1280, width_coefficient,
                          depth_divisor, min_depth, skip),
            kernel_size=1,
            stride=1,
            bias_attr=False)
        self._norm = nn.BatchNorm2D(
            round_filters(self.feature_size or 1280, width_coefficient,
                          depth_divisor, min_depth, skip),
            self.bn_momentum,
            self.bn_epsilon,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self._act = activation_fn(act_fn)

        self._avg_pooling = nn.AdaptiveAvgPool2D(output_size=1)

        if isinstance(self.dropout_rate,
                      (list, tuple)) or self.dropout_rate > 0:
            self._dropout = nn.Dropout(self.dropout_rate[0] if isinstance(
                self.dropout_rate, (list, tuple)) else self.dropout_rate)
        else:
            self._dropout = None

    def forward(self, x):
        """Call the layer."""
        outputs = self._act(self._norm(self._conv_head(x)))

        if self._local_pooling:
            outputs = F.adaptive_avg_pool2d(outputs, output_size=1)
            if self._dropout:
                outputs = self._dropout(outputs)
            if self._fc:
                outputs = paddle.squeeze(outputs, axis=[2, 3])
                outputs = self._fc(outputs)
        else:
            outputs = self._avg_pooling(outputs)
            if self._dropout:
                outputs = self._dropout(outputs)
        return paddle.flatten(outputs, start_axis=1)


class EfficientNetV2(nn.Layer):
    """A class implements tf.keras.Model.

        Reference: https://arxiv.org/abs/1807.11626
    """

    def __init__(self,
                 model_name,
                 blocks_args=None,
                 mconfig=None,
                 include_top=True,
                 class_num=1000,
                 padding_type="SAME"):
        """Initializes an `Model` instance.

        Args:
            model_name: A string of model name.
            model_config: A dict of model configurations or a string of hparams.
        Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super(EfficientNetV2, self).__init__()
        self.blocks_args = blocks_args
        self.mconfig = mconfig
        """Builds a model."""
        self._blocks = nn.LayerList()

        cur_stage = 0
        # Stem part.
        self._stem = Stem(
            self.mconfig.width_coefficient,
            self.mconfig.depth_divisor,
            self.mconfig.min_depth,
            False,
            self.mconfig.bn_momentum,
            self.mconfig.bn_epsilon,
            self.mconfig.act_fn,
            stem_channels=self.blocks_args[0].in_channels,
            cur_stage=cur_stage,
            padding_type=padding_type,
            model_name=model_name)
        cur_stage += 1

        # Builds blocks.
        for block_args in self.blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            in_channels = round_filters(
                block_args.in_channels, self.mconfig.width_coefficient,
                self.mconfig.depth_divisor, self.mconfig.min_depth, False)
            out_channels = round_filters(
                block_args.out_channels, self.mconfig.width_coefficient,
                self.mconfig.depth_divisor, self.mconfig.min_depth, False)

            repeats = round_repeats(block_args.num_repeat,
                                    self.mconfig.depth_coefficient)
            block_args.update(
                dict(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    num_repeat=repeats))

            # The first block needs to take care of stride and filter size increase.
            conv_block = {
                0: MBConvBlock,
                1: FusedMBConvBlock
            }[block_args.conv_type]
            self._blocks.append(
                conv_block(block_args.se_ratio, block_args.in_channels,
                           block_args.expand_ratio, block_args.kernel_size,
                           block_args.strides, block_args.out_channels,
                           self.mconfig.bn_momentum, self.mconfig.bn_epsilon,
                           self.mconfig.local_pooling, self.mconfig.
                           conv_dropout, cur_stage, padding_type, model_name))
            if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                block_args.in_channels = block_args.out_channels
                block_args.strides = 1
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(
                    conv_block(
                        block_args.se_ratio, block_args.in_channels,
                        block_args.expand_ratio, block_args.kernel_size,
                        block_args.strides, block_args.out_channels,
                        self.mconfig.bn_momentum, self.mconfig.bn_epsilon,
                        self.mconfig.local_pooling, self.mconfig.conv_dropout,
                        cur_stage, padding_type, model_name))
            cur_stage += 1

        # Head part.
        self._head = Head(
            self.blocks_args[-1].out_channels, self.mconfig.feature_size,
            self.mconfig.bn_momentum, self.mconfig.bn_epsilon,
            self.mconfig.act_fn, self.mconfig.dropout_rate,
            self.mconfig.local_pooling, self.mconfig.width_coefficient,
            self.mconfig.depth_divisor, self.mconfig.min_depth, False)

        # top part for classification
        if include_top and class_num:
            self._fc = nn.Linear(
                self.mconfig.feature_size,
                class_num,
                bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        else:
            self._fc = None

        # initialize weight
        def _init_weights(m):
            if isinstance(m, nn.Conv2D):
                out_filters, in_channels, kernel_height, kernel_width = m.weight.shape
                if in_channels == 1 and out_filters > in_channels:
                    out_filters = in_channels
                fan_out = int(kernel_height * kernel_width * out_filters)
                Normal(mean=0.0, std=np.sqrt(2.0 / fan_out))(m.weight)
            elif isinstance(m, nn.Linear):
                init_range = 1.0 / np.sqrt(m.weight.shape[1])
                Uniform(-init_range, init_range)(m.weight)
                Constant(0.0)(m.bias)

        self.apply(_init_weights)

    def forward(self, inputs):
        # Calls Stem layers
        outputs = self._stem(inputs)
        # print(f"stem: {outputs.mean().item():.10f}")

        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            survival_prob = self.mconfig.survival_prob
            if survival_prob:
                drop_rate = 1.0 - survival_prob
                survival_prob = 1.0 - drop_rate * float(idx) / len(
                    self._blocks)
            outputs = block(outputs, survival_prob=survival_prob)

        # Head to obtain the final feature.
        outputs = self._head(outputs)
        # Calls final dense layers and returns logits.
        if self._fc:
            outputs = self._fc(outputs)

        return outputs


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def EfficientNetV2_S(include_top=True, pretrained=False, **kwargs):
    """Get a V2 model instance.

    Returns:
        nn.Layer: A single model instantce
    """
    model_name = "efficientnetv2-s"
    model_config = efficientnetv2_config(model_name)
    model = EfficientNetV2(model_name, model_config.model.blocks_args,
                           model_config.model, include_top, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetV2_S"])
    return model


def EfficientNetV2_M(include_top=True, pretrained=False, **kwargs):
    """Get a V2 model instance.

    Returns:
        nn.Layer: A single model instantce
    """
    model_name = "efficientnetv2-m"
    model_config = efficientnetv2_config(model_name)
    model = EfficientNetV2(model_name, model_config.model.blocks_args,
                           model_config.model, include_top, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetV2_M"])
    return model


def EfficientNetV2_L(include_top=True, pretrained=False, **kwargs):
    """Get a V2 model instance.

    Returns:
        nn.Layer: A single model instantce
    """
    model_name = "efficientnetv2-l"
    model_config = efficientnetv2_config(model_name)
    model = EfficientNetV2(model_name, model_config.model.blocks_args,
                           model_config.model, include_top, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetV2_L"])
    return model


def EfficientNetV2_XL(include_top=True, pretrained=False, **kwargs):
    """Get a V2 model instance.

    Returns:
        nn.Layer: A single model instantce
    """
    model_name = "efficientnetv2-xl"
    model_config = efficientnetv2_config(model_name)
    model = EfficientNetV2(model_name, model_config.model.blocks_args,
                           model_config.model, include_top, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientNetV2_XL"])
    return model
