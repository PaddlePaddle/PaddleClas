# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/2404.10518

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn import BatchNorm, Conv2D, Dropout, Linear, Identity, Flatten
from paddle.regularizer import L2Decay

from .custom_devices_layers import AdaptiveAvgPool2D
from ..base.theseus_layer import TheseusLayer
from ..model_zoo.vision_transformer import DropPath
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileNetV4_conv_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_large_pretrained.pdparams",
    "MobileNetV4_conv_medium":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_medium_pretrained.pdparams",
    "MobileNetV4_conv_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_conv_small_pretrained.pdparams",
    "MobileNetV4_hybrid_large":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_hybrid_large_pretrained.pdparams",
    "MobileNetV4_hybrid_medium":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/MobileNetV4_hybrid_medium_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())
STEM_CONV_NUMBER = 32
LAST_CONV = 1280

NET_CONFIG = {
    "conv_small": [
        # stage 0, 112x112 
        # type, out, kernal_size, act, stride
        ["cn", 32, 3, "relu", 2],
        ["cn", 32, 1, "relu", 1],
        # stage 1, 56x56
        ["cn", 96, 3, "relu", 2],
        ["cn", 64, 1, "relu", 1],
        # stage 2, 28x28
        # type, out, mid_c, first_kernal_size, mid_kernal_size, act, stride
        ["uir", 96, 192, 5, 5, "relu", 2],
        ["uir", 96, 192, 0, 3, "relu", 1],
        ["uir", 96, 192, 0, 3, "relu", 1],
        ["uir", 96, 192, 0, 3, "relu", 1],
        ["uir", 96, 192, 0, 3, "relu", 1],
        ["uir", 96, 384, 3, 0, "relu", 1],
        # stage 3,  14x14
        ["uir", 128, 576, 3, 3, "relu", 2],
        ["uir", 128, 512, 5, 5, "relu", 1],
        ["uir", 128, 512, 0, 5, "relu", 1],
        ["uir", 128, 384, 0, 5, "relu", 1],
        ["uir", 128, 512, 0, 3, "relu", 1],
        ["uir", 128, 512, 0, 3, "relu", 1],
        # stage 4, 7x7
        ["cn", 960, 1, "relu", 1],
    ],
    "conv_medium": [
        # stage 0, 112x112 
        ["er", 48, 128, 3, "relu", 2],
        # stage 1, 56x56
        ["uir", 80, 192, 3, 5, "relu", 2],
        ["uir", 80, 160, 3, 3, "relu", 1],
        # stage 2, 28x28
        ["uir", 160, 480, 3, 5, "relu", 2],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["uir", 160, 640, 3, 5, "relu", 1],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["uir", 160, 640, 3, 0, "relu", 1],
        ["uir", 160, 320, 0, 0, "relu", 1],
        ["uir", 160, 640, 3, 0, "relu", 1],
        # stage 3,  14x14
        ["uir", 256, 960, 5, 5, "relu", 2],
        ["uir", 256, 1024, 5, 5, "relu", 1],
        ["uir", 256, 1024, 3, 5, "relu", 1],
        ["uir", 256, 1024, 3, 5, "relu", 1],
        ["uir", 256, 1024, 0, 0, "relu", 1],
        ["uir", 256, 1024, 3, 0, "relu", 1],
        ["uir", 256, 512, 3, 5, "relu", 1],
        ["uir", 256, 1024, 5, 5, "relu", 1],
        ["uir", 256, 1024, 0, 0, "relu", 1],
        ["uir", 256, 1024, 0, 0, "relu", 1],
        ["uir", 256, 512, 5, 0, "relu", 1],
        # stage 4, 7x7
        ["cn", 960, 1, "relu", 1],
    ],
    "conv_large": [
        # stem_size = 24
        ["er", 48, 96, 3, "relu", 2],
        # stage 1, 56x56
        ["uir", 96, 192, 3, 5, "relu", 2],
        ["uir", 96, 384, 3, 3, "relu", 1],
        # stage 2, 28x28 in
        ["uir", 192, 384, 3, 5, "relu", 2],
        ["uir", 192, 768, 3, 3, "relu", 1],
        ["uir", 192, 768, 3, 3, "relu", 1],
        ["uir", 192, 768, 3, 3, "relu", 1],
        ["uir", 192, 768, 3, 5, "relu", 1],
        ["uir", 192, 768, 5, 3, "relu", 1],
        ["uir", 192, 768, 5, 3, "relu", 1],
        ["uir", 192, 768, 5, 3, "relu", 1],
        ["uir", 192, 768, 5, 3, "relu", 1],
        ["uir", 192, 768, 5, 3, "relu", 1],
        ["uir", 192, 768, 3, 0, "relu", 1],
        # stage 3,  14x14 in
        ["uir", 512, 768, 5, 5, "relu", 2],
        ["uir", 512, 2048, 5, 5, "relu", 1],
        ["uir", 512, 2048, 5, 5, "relu", 1],
        ["uir", 512, 2048, 5, 5, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        ["uir", 512, 2048, 5, 3, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        ["uir", 512, 2048, 5, 3, "relu", 1],
        ["uir", 512, 2048, 5, 5, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        ["uir", 512, 2048, 5, 0, "relu", 1],
        # stage 4, 7x7
        ["cn", 960, 1, "relu", 1],
    ],
    "hybrid_medium": [
        # stem_size = 32
        ["er", 48, 128, 3, "relu", 2],
        # stage 1, 56x56
        ["uir", 80, 192, 3, 5, "relu", 2],
        ["uir", 80, 160, 3, 3, "relu", 1],
        # stage 2, 28x28
        ["uir", 160, 480, 3, 5, "relu", 2],
        ["uir", 160, 320, 0, 0, "relu", 1],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["uir", 160, 640, 3, 5, "relu", 1],
        # type, out, kv_dim, kernal_size, kv_stride, act, stride
        ["mqa", 160, 64, 3, 4, 2, "relu", 1],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["mqa", 160, 64, 3, 4, 2, "relu", 1],
        ["uir", 160, 640, 3, 0, "relu", 1],
        ["mqa", 160, 64, 3, 4, 2, "relu", 1],
        ["uir", 160, 640, 3, 3, "relu", 1],
        ["mqa", 160, 64, 3, 4, 2, "relu", 1],
        ["uir", 160, 640, 3, 0, "relu", 1],
        # stage 3,  14x14
        ["uir", 256, 960, 5, 5, "relu", 2],
        ["uir", 256, 1024, 5, 5, "relu", 1],
        ["uir", 256, 1024, 3, 5, "relu", 1],
        ["uir", 256, 1024, 3, 5, "relu", 1],
        ["uir", 256, 512, 0, 0, "relu", 1],
        ["uir", 256, 512, 3, 5, "relu", 1],
        ["uir", 256, 512, 0, 0, "relu", 1],
        ["uir", 256, 1024, 0, 0, "relu", 1],
        ["mqa", 256, 64, 3, 4, 1, "relu", 1],
        ["uir", 256, 1024, 3, 0, "relu", 1],
        ["mqa", 256, 64, 3, 4, 1, "relu", 1],
        ["uir", 256, 1024, 5, 5, "relu", 1],
        ["mqa", 256, 64, 3, 4, 1, "relu", 1],
        ["uir", 256, 1024, 5, 0, "relu", 1],
        ["mqa", 256, 64, 3, 4, 1, "relu", 1],
        ["uir", 256, 1024, 5, 0, "relu", 1],
        # stage 4, 7x7
        ["cn", 960, 1, "relu", 1],
    ],
    "hybrid_large": [
        # stem_size = 24
        ["er", 48, 96, 3, "gelu", 2],
        # stage 1, 56x56
        ["uir", 96, 192, 3, 5, "gelu", 2],
        ["uir", 96, 384, 3, 3, "gelu", 1],
        # stage 2, 28x28 in
        ["uir", 192, 384, 3, 5, "gelu", 2],
        ["uir", 192, 768, 3, 3, "gelu", 1],
        ["uir", 192, 768, 3, 3, "gelu", 1],
        ["uir", 192, 768, 3, 3, "gelu", 1],
        ["uir", 192, 768, 3, 5, "gelu", 1],
        ["uir", 192, 768, 5, 3, "gelu", 1],
        ["uir", 192, 768, 5, 3, "gelu", 1],
        ["mqa", 192, 48, 3, 8, 2, "gelu", 1],
        ["uir", 192, 768, 5, 3, "gelu", 1],
        ["mqa", 192, 48, 3, 8, 2, "gelu", 1],
        ["uir", 192, 768, 5, 3, "gelu", 1],
        ["mqa", 192, 48, 3, 8, 2, "gelu", 1],
        ["uir", 192, 768, 5, 3, "gelu", 1],
        ["mqa", 192, 48, 3, 8, 2, "gelu", 1],
        ["uir", 192, 768, 3, 0, "gelu", 1],
        # stage 3,  14x14
        ["uir", 512, 768, 5, 5, "gelu", 2],
        ["uir", 512, 2048, 5, 5, "gelu", 1],
        ["uir", 512, 2048, 5, 5, "gelu", 1],
        ["uir", 512, 2048, 5, 5, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["uir", 512, 2048, 5, 3, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["uir", 512, 2048, 5, 3, "gelu", 1],
        ["uir", 512, 2048, 5, 5, "gelu", 1],
        ["mqa", 512, 64, 3, 8, 1, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["mqa", 512, 64, 3, 8, 1, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["mqa", 512, 64, 3, 8, 1, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        ["mqa", 512, 64, 3, 8, 1, "gelu", 1],
        ["uir", 512, 2048, 5, 0, "gelu", 1],
        # stage 4, 7x7
        ["cn", 960, 1, "gelu", 1],
    ]
}

MODEL_STAGES_PATTERN = {
    "MobileNetV4_conv_small":
    ["blocks[1]", "blocks[3]", "blocks[9]", "blocks[15]", "blocks[16]"],
    "MobileNetV4_conv_medium":
    ["blocks[0]", "blocks[2]", "blocks[10]", "blocks[21]", "blocks[22]"],
    "MobileNetV4_conv_large":
    ["blocks[0]", "blocks[2]", "blocks[13]", "blocks[26]", "blocks[27]"],
    "MobileNetV4_hybrid_medium":
    ["blocks[0]", "blocks[2]", "blocks[14]", "blocks[30]", "blocks[31]"],
    "MobileNetV4_hybrid_large":
    ["blocks[0]", "blocks[2]", "blocks[17]", "blocks[35]", "blocks[36]"],
}


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _create_act(act):
    if act == "hardswish":
        return nn.Hardswish()
    elif act == "relu":
        return nn.ReLU()
    elif act == "gelu":
        return nn.GELU(approximate=False)
    elif act is None:
        return None
    else:
        raise RuntimeError(
            "The activation function is not supported: {}".format(act))


class ConvBnAct(TheseusLayer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size,
                 stride,
                 padding,
                 num_groups=1,
                 drop_path_rate=0.0,
                 if_act=True,
                 act=None):
        super().__init__()

        self.drop_path_rate = drop_path_rate
        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=num_groups,
            bias_attr=False)
        self.bn = BatchNorm(
            num_channels=out_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.if_act = if_act
        if self.if_act:
            self.act = _create_act(act)
        if self.drop_path_rate > 0:
            self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        if self.drop_path_rate > 0:
            x = self.drop_path(x)
        return x


class EdgeResidual(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stride,
                 drop_path_rate=0.0,
                 if_act=False,
                 act=None):
        super(EdgeResidual, self).__init__()

        self.if_shortcut = stride == 1 and in_c == out_c
        self.conv_exp = ConvBnAct(
            in_c=in_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=int((filter_size - 1) // 2),
            if_act=True,
            act=act)
        self.conv_pwl = ConvBnAct(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=act)
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate else Identity()

    def forward(self, x):
        identity = x
        x = self.conv_exp(x)
        x = self.conv_pwl(x)
        if self.if_shortcut:
            x = paddle.add(identity, self.drop_path(x))
        return x


class UniversalInvertedResidual(TheseusLayer):
    def __init__(self,
                 in_c,
                 mid_c,
                 out_c,
                 filter_size,
                 stem_kernel_size=None,
                 stride=1,
                 drop_path_rate=0.0,
                 layer_scale_init_value=0.0,
                 if_act=False,
                 act=None):
        super().__init__()

        self.if_shortcut = stride == 1 and in_c == out_c
        self.layer_scale_init_value = layer_scale_init_value
        if stem_kernel_size:
            self.dw_start = ConvBnAct(
                in_c=in_c,
                out_c=in_c,
                filter_size=stem_kernel_size,
                stride=1,
                padding=int((stem_kernel_size - 1) // 2),
                num_groups=in_c,
                if_act=False,
                act=None)
        else:
            self.dw_start = Identity()

        self.pw_exp = ConvBnAct(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            num_groups=1,
            if_act=True,
            act=act)

        if filter_size:
            self.dw_mid = ConvBnAct(
                in_c=mid_c,
                out_c=mid_c,
                filter_size=filter_size,
                stride=stride,
                padding=int((filter_size - 1) // 2),
                num_groups=mid_c,
                if_act=True,
                act=act)
        else:
            self.dw_mid = Identity()
        self.pw_proj = ConvBnAct(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=False,
            act=None)
        if layer_scale_init_value > 0.0:
            self.layer_scale = LayerScale2D(out_c, layer_scale_init_value)

        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate else Identity()

    def forward(self, x):
        identity = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.pw_proj(x)
        if self.layer_scale_init_value > 0.0:
            x = self.layer_scale(x)
        if self.if_shortcut:
            x = paddle.add(identity, self.drop_path(x))
        return x


class LayerScale2D(nn.Layer):
    def __init__(self, dim, init_values=1e-05, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        gamma = self.gamma.reshape([1, -1, 1, 1])
        return (x.multiply_(y=paddle.to_tensor(gamma))
                if self.inplace else x * gamma)


class MobileAttention(nn.Layer):
    def __init__(self,
                 in_c,
                 out_c,
                 filter_size=3,
                 stride=1,
                 num_head=4,
                 query_dim=256,
                 kv_dim=64,
                 kv_stride=1,
                 drop_path_rate=0.0,
                 attn_drop_rate=0.0,
                 dropout_prob=0.0,
                 layer_scale_init_value=0.0,
                 if_act=True,
                 act=None,
                 use_fused_attn=False):
        super(MobileAttention, self).__init__()

        self.if_shortcut = stride == 1 and in_c == out_c
        self.kv_stride = kv_stride
        self.kv_dim = kv_dim
        self.num_head = num_head
        self.query_dim = query_dim
        self.attn_drop_rate = attn_drop_rate
        self.use_fused_attn = use_fused_attn
        self.norm = BatchNorm(
            num_channels=in_c,
            act=None,
            param_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.query_proj = Conv2D(
            in_channels=in_c,
            out_channels=query_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        if kv_stride > 1:
            self.key_down_proj = ConvBnAct(
                in_c=in_c,
                out_c=in_c,
                filter_size=filter_size,
                stride=kv_stride,
                padding=int((filter_size - 1) // 2),
                num_groups=in_c,
                if_act=False)
            self.value_down_proj = ConvBnAct(
                in_c=in_c,
                out_c=in_c,
                filter_size=filter_size,
                stride=kv_stride,
                padding=int((filter_size - 1) // 2),
                num_groups=in_c,
                if_act=False)
        self.key_proj = Conv2D(
            in_channels=in_c,
            out_channels=kv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        self.value_proj = Conv2D(
            in_channels=in_c,
            out_channels=kv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        self.proj = Conv2D(
            in_channels=query_dim,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False)
        if not self.use_fused_attn:
            self.scale = query_dim**-0.5
            self.softmax = nn.Softmax(-1)
            self.attn_drop = Dropout(self.attn_drop_rate)
        self.drop = Dropout(dropout_prob)
        self.layer_scale_init_value = layer_scale_init_value
        if layer_scale_init_value > 0.0:
            self.layer_scale = LayerScale2D(out_c, layer_scale_init_value)
        self.drop_path = (DropPath(drop_path_rate)
                          if drop_path_rate else Identity())

    def forward(self, x, attn_mask=None):
        identity = x
        x = self.norm(x)
        B, C, H, W = tuple(x.shape)
        q = self.query_proj(x).reshape(
            [B, self.num_head, self.query_dim // self.num_head, H * W])
        q = q.transpose([0, 3, 1, 2])
        if self.kv_stride > 1:
            k = self.key_proj(self.key_down_proj(x))
            v = self.value_proj(self.value_down_proj(x))
        else:
            k = self.key_proj(x)
            v = self.value_proj(x)
        k = k.reshape(
            [B, self.kv_dim, 1, H // self.kv_stride * W // self.kv_stride])
        k = k.transpose([0, 3, 2, 1])
        v = v.reshape(
            [B, self.kv_dim, 1, H // self.kv_stride * W // self.kv_stride])
        v = v.transpose([0, 3, 2, 1])
        if self.use_fused_attn:
            attn = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop_rate if self.training else 0.0)
        else:
            q = q.transpose([0, 2, 1, 3]) * self.scale
            v = v.transpose([0, 2, 1, 3])
            attn = q @ k.transpose([0, 2, 3, 1])
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            attn = attn @ v
            attn = attn.transpose([0, 2, 1, 3])

        attn = attn.reshape([B, H, W, self.query_dim])
        x = self.proj(attn.transpose([0, 3, 1, 2]))
        x = self.drop(x)
        if self.layer_scale_init_value > 0.0:
            x = self.layer_scale(x)
        if self.if_shortcut:
            x = paddle.add(identity, self.drop_path(x))
        return x


class MobileNetV4(TheseusLayer):
    """
    MobileNetV4
    Args:
        config: list. MobileNetV4 depthwise blocks config.
        stages_pattern: list. The pattern of each stage blocks.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
        class_num: int=1000. The number of classes.
        inplanes: int=32. The output channel number of first convolution layer.
        act: str="relu". The activation function.
        class_expand: int=960. The output channel number of last convolution layer. 
        drop_path_rate: float=0.0. Probability of dropping path.
        drop_rate: float=0.0.  Probability of setting units to zero.
    Returns:
        model: nn.Layer. Specific MobileNetV4 model depends on args.
    """
    def __init__(self,
                 config,
                 stages_pattern,
                 scale=1.0,
                 class_num=1000,
                 inplanes=STEM_CONV_NUMBER,
                 class_expand=LAST_CONV,
                 act="relu",
                 drop_path_rate=0.0,
                 drop_rate=0.0,
                 layer_scale_init_value=0.0,
                 return_patterns=None,
                 return_stages=None,
                 use_fused_attn=False,
                 **kwargs):
        super(MobileNetV4, self).__init__()
        self.cfg = config
        self.scale = scale
        self.drop_path_rate = drop_path_rate
        self.inplanes = inplanes
        self.class_expand = class_expand
        self.class_num = class_num
        self.conv_stem = ConvBnAct(
            in_c=3,
            out_c=_make_divisible(self.inplanes * self.scale),
            filter_size=3,
            stride=2,
            padding=1,
            if_act=True,
            act=act)

        blocks = []
        block_count = len(self.cfg)
        for i in range(block_count):
            type = self.cfg[i][0]
            if type == "cn":
                _, exp, k, act, s = self.cfg[i]
                block = ConvBnAct(
                    in_c=_make_divisible(self.inplanes * self.scale if i == 0
                                         else self.cfg[i - 1][1] * self.scale),
                    out_c=_make_divisible(exp * self.scale),
                    filter_size=k,
                    stride=s,
                    padding=int((k - 1) // 2),
                    num_groups=1,
                    drop_path_rate=self.drop_path_rate * i / block_count,
                    if_act=True,
                    act=act)
            elif type == "uir":
                _, c, exp, k_start, k, act, s = self.cfg[i]
                block = UniversalInvertedResidual(
                    in_c=_make_divisible(self.inplanes * self.scale if i == 0
                                         else self.cfg[i - 1][1] * self.scale),
                    mid_c=_make_divisible(self.scale * exp),
                    out_c=_make_divisible(self.scale * c),
                    filter_size=k,
                    stem_kernel_size=k_start,
                    stride=s,
                    drop_path_rate=self.drop_path_rate * i / block_count,
                    layer_scale_init_value=layer_scale_init_value,
                    if_act=True,
                    act=act)
            elif type == "er":
                _, c, exp, k, act, s = self.cfg[i]
                block = EdgeResidual(
                    in_c=_make_divisible(self.inplanes * self.scale if i == 0
                                         else self.cfg[i - 1][1] * self.scale),
                    mid_c=_make_divisible(self.scale * exp),
                    out_c=_make_divisible(self.scale * c),
                    filter_size=k,
                    stride=s,
                    drop_path_rate=self.drop_path_rate * i / block_count,
                    if_act=True,
                    act=act)
            elif type == "mqa":
                # type, out,kv_dim, kernal_size, kv_stride, act, stride
                _, c, dim, k, head, kv_stride, act, s = self.cfg[i]
                block = MobileAttention(
                    in_c=_make_divisible(self.inplanes * self.scale if i == 0
                                         else self.cfg[i - 1][1] * self.scale),
                    out_c=_make_divisible(self.scale * c),
                    filter_size=k,
                    stride=s,
                    num_head=head,
                    query_dim=_make_divisible(self.scale * head * dim),
                    kv_dim=_make_divisible(self.scale * dim),
                    kv_stride=kv_stride,
                    drop_path_rate=self.drop_path_rate * i / block_count,
                    layer_scale_init_value=layer_scale_init_value,
                    if_act=True,
                    act=act,
                    use_fused_attn=use_fused_attn)
            blocks.append(block)
        self.blocks = nn.Sequential(*blocks)
        self.global_pool = AdaptiveAvgPool2D(1)
        self.conv_head = ConvBnAct(
            in_c=_make_divisible(self.scale * self.cfg[-1][1]),
            out_c=self.class_expand,
            filter_size=1,
            stride=1,
            padding=0,
            if_act=True,
            act=act)
        self.flatten = Flatten(start_axis=1, stop_axis=-1)
        self.dropout = Dropout(drop_rate)
        self.classifier = Linear(self.class_expand,
                                 class_num) if class_num > 0 else Identity()
        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.flatten(x)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError("pretrained type is not available. ")


def MobileNetV4_conv_small(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV4_conv_small
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV4_conv_small` model depends on args.
    """
    model = MobileNetV4(
        config=NET_CONFIG["conv_small"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV4_conv_small"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_small"],
                     use_ssld)
    return model


def MobileNetV4_conv_medium(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV4_conv_medium
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV4_conv_medium` model depends on args.
    """
    model = MobileNetV4(
        config=NET_CONFIG["conv_medium"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV4_conv_medium"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_medium"],
                     use_ssld)
    return model


def MobileNetV4_conv_large(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV4_conv_large
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV4_conv_large` model depends on args.
    """
    model = MobileNetV4(
        config=NET_CONFIG["conv_large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV4_conv_large"],
        inplanes=24,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_conv_large"],
                     use_ssld)
    return model


def MobileNetV4_hybrid_medium(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV4_hybrid_medium
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV4_hybrid_medium` model depends on args.
    """
    model = MobileNetV4(
        config=NET_CONFIG["hybrid_medium"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV4_hybrid_medium"],
        layer_scale_init_value=1e-05,
        **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["MobileNetV4_hybrid_medium"], use_ssld)
    return model


def MobileNetV4_hybrid_large(pretrained=False, use_ssld=False, **kwargs):
    """
    MobileNetV4_hybrid_large
    Args:
        pretrained: bool=False or str. If `True` load pretrained parameters, `False` otherwise.
                    If str, means the path of the pretrained model.
        use_ssld: bool=False. Whether using distillation pretrained model when pretrained=True.
    Returns:
        model: nn.Layer. Specific `MobileNetV4_hybrid_large` model depends on args.
    """
    model = MobileNetV4(
        config=NET_CONFIG["hybrid_large"],
        scale=1.0,
        stages_pattern=MODEL_STAGES_PATTERN["MobileNetV4_hybrid_large"],
        inplanes=24,
        act="gelu",
        layer_scale_init_value=1e-05,
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV4_hybrid_large"],
                     use_ssld)
    return model
