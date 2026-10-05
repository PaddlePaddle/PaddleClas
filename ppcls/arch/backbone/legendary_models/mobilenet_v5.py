# copyright (c) 2026 PaddlePaddle Authors. All Rights Reserve.
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

# reference:
# https://github.com/huggingface/pytorch-image-models/blob/8d0f79effa3dbc922afbfb431fbadd4648938de7/timm/models/mobilenetv5.py

from __future__ import absolute_import, division, print_function

import re

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn import Dropout, Flatten, Identity, Linear
from paddle.nn import Conv2D

from .custom_devices_layers import AdaptiveAvgPool2D
from .mobilenet_v4 import LayerScale2D, _make_divisible
from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileNetV5_300M": "",
    "MobileNetV5_base": "",
    "MobileNetV5_300M_enc": "",
    "MobileNetV5_300M_enc_cls": "",
}

__all__ = list(MODEL_URLS.keys()) + ["MobileNetV5_300m"]

_ARCH_DEFS = {
    "mobilenetv5_300m": [
        [
            "er_r1_k3_s2_e4_c128",
            "er_r1_k3_s1_e4_c128",
            "er_r1_k3_s1_e4_c128",
        ],
        [
            "uir_r1_a3_k5_s2_e6_c256",
            "uir_r1_a5_k0_s1_e4_c256",
            "uir_r1_a3_k0_s1_e4_c256",
            "uir_r1_a5_k0_s1_e4_c256",
            "uir_r1_a3_k0_s1_e4_c256",
        ],
        [
            "uir_r1_a5_k5_s2_e6_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a5_k0_s1_e4_c640",
            "uir_r1_a0_k0_s1_e1_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
            "mqa_r1_k3_h12_v2_s1_d64_c640",
            "uir_r1_a0_k0_s1_e2_c640",
        ],
        [
            "uir_r1_a5_k5_s2_e6_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
            "mqa_r1_k3_h16_s1_d96_c1280",
            "uir_r1_a0_k0_s1_e2_c1280",
        ],
    ],
    "mobilenetv5_base": [
        [
            "er_r1_k3_s2_e4_c128",
            "er_r1_k3_s1_e4_c128",
            "er_r1_k3_s1_e4_c128",
        ],
        [
            "uir_r1_a3_k5_s2_e6_c256",
            "uir_r1_a5_k0_s1_e4_c256",
            "uir_r1_a3_k0_s1_e4_c256",
            "uir_r1_a5_k0_s1_e4_c256",
            "uir_r1_a3_k0_s1_e4_c256",
        ],
        [
            "uir_r1_a5_k5_s2_e6_c512",
            "uir_r1_a5_k0_s1_e4_c512",
            "uir_r1_a5_k0_s1_e4_c512",
            "uir_r1_a0_k0_s1_e1_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
            "mqa_r1_k3_h8_s2_d64_c512",
            "uir_r1_a0_k0_s1_e2_c512",
        ],
        [
            "uir_r1_a5_k5_s2_e6_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
            "mqa_r1_k3_h16_s1_d64_c1024",
            "uir_r1_a0_k0_s1_e2_c1024",
        ],
    ],
}


def _parse_block(block_str):
    block_type, *parts = block_str.split("_")
    params = {}
    for p in parts:
        m = re.match(r"([a-z]+)([-0-9.]+)", p)
        if m:
            key, val = m.groups()
            params[key] = float(val) if "." in val else int(val)
    return block_type, params


def _get_padding(kernel_size, pad_type):
    if pad_type == "same":
        return "SAME"
    return int((kernel_size - 1) // 2)


class RmsNorm2D(nn.Layer):

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[num_channels], default_initializer=nn.initializer.Constant(1.0)
        )

    def forward(self, x):
        mean_square = paddle.mean(x * x, axis=1, keepdim=True)
        x = x * paddle.rsqrt(mean_square + self.eps)
        w = self.weight.reshape([1, -1, 1, 1])
        return x * w


class ConvBnActV5(nn.Layer):

    def __init__(
        self,
        in_c,
        out_c,
        filter_size,
        stride,
        padding,
        groups=1,
        conv_bias=False,
        if_act=True,
        act="gelu",
    ):
        super().__init__()
        self.conv = Conv2D(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=filter_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=conv_bias,
        )
        self.bn = RmsNorm2D(out_c)
        self.if_act = if_act
        if self.if_act:
            if act == "gelu":
                self.act = nn.GELU(approximate=True)
            elif act == "relu":
                self.act = nn.ReLU()
            else:
                self.act = Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.if_act:
            x = self.act(x)
        return x


class UniversalInvertedResidualV5(nn.Layer):

    def __init__(
        self,
        in_c,
        mid_c,
        out_c,
        filter_size=0,
        stem_kernel_size=0,
        stride=1,
        pad_type="",
        layer_scale_init_value=None,
        act="gelu",
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        if stem_kernel_size and stem_kernel_size > 0:
            self.dw_start = ConvBnActV5(
                in_c=in_c,
                out_c=in_c,
                filter_size=stem_kernel_size,
                stride=1,
                padding=_get_padding(stem_kernel_size, pad_type),
                groups=in_c,
                conv_bias=False,
                if_act=False,
                act=act,
            )
        else:
            self.dw_start = Identity()
        self.pw_exp = ConvBnActV5(
            in_c=in_c,
            out_c=mid_c,
            filter_size=1,
            stride=1,
            padding=0,
            conv_bias=False,
            if_act=True,
            act=act,
        )
        if filter_size and filter_size > 0:
            self.dw_mid = ConvBnActV5(
                in_c=mid_c,
                out_c=mid_c,
                filter_size=filter_size,
                stride=stride,
                padding=_get_padding(filter_size, pad_type),
                groups=mid_c,
                conv_bias=False,
                if_act=True,
                act=act,
            )
        else:
            self.dw_mid = Identity()
        self.pw_proj = ConvBnActV5(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            conv_bias=False,
            if_act=False,
            act=act,
        )
        self.layer_scale = (
            LayerScale2D(out_c, layer_scale_init_value)
            if layer_scale_init_value is not None
            else None
        )

    def forward(self, x):
        identity = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.pw_proj(x)
        if self.layer_scale is not None:
            x = self.layer_scale(x)
        if self.if_shortcut:
            x = x + identity
        return x


class EdgeResidualV5(nn.Layer):

    def __init__(
        self, in_c, mid_c, out_c, filter_size, stride=1, pad_type="", act="gelu"
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.conv_exp = ConvBnActV5(
            in_c=in_c,
            out_c=mid_c,
            filter_size=filter_size,
            stride=stride,
            padding=_get_padding(filter_size, pad_type),
            conv_bias=False,
            if_act=True,
            act=act,
        )
        self.conv_pwl = ConvBnActV5(
            in_c=mid_c,
            out_c=out_c,
            filter_size=1,
            stride=1,
            padding=0,
            conv_bias=False,
            if_act=False,
            act=act,
        )

    def forward(self, x):
        identity = x
        x = self.conv_exp(x)
        x = self.conv_pwl(x)
        if self.if_shortcut:
            x = x + identity
        return x


class MobileAttentionV5(nn.Layer):

    def __init__(
        self,
        in_c,
        out_c,
        filter_size=3,
        stride=1,
        num_head=8,
        query_dim=256,
        kv_dim=64,
        kv_stride=1,
        pad_type="",
        drop_path_rate=0.0,
        layer_scale_init_value=1e-5,
        **kwargs,
    ):
        super().__init__()
        self.if_shortcut = stride == 1 and in_c == out_c
        self.kv_stride = kv_stride
        self.kv_dim = kv_dim
        self.num_head = num_head
        self.query_dim = query_dim

        self.norm = RmsNorm2D(in_c)
        self.query_proj = Conv2D(
            in_channels=in_c,
            out_channels=query_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False,
        )
        if kv_stride > 1:
            self.key_down_proj = ConvBnActV5(
                in_c=in_c,
                out_c=in_c,
                filter_size=filter_size,
                stride=kv_stride,
                padding=_get_padding(filter_size, pad_type),
                groups=in_c,
                conv_bias=False,
                if_act=False,
                act="gelu",
            )
            self.value_down_proj = ConvBnActV5(
                in_c=in_c,
                out_c=in_c,
                filter_size=filter_size,
                stride=kv_stride,
                padding=_get_padding(filter_size, pad_type),
                groups=in_c,
                conv_bias=False,
                if_act=False,
                act="gelu",
            )
        self.key_proj = Conv2D(
            in_channels=in_c,
            out_channels=kv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False,
        )
        self.value_proj = Conv2D(
            in_channels=in_c,
            out_channels=kv_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False,
        )
        self.proj = Conv2D(
            in_channels=query_dim,
            out_channels=out_c,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias_attr=False,
        )

        # timm MultiQueryAttention2d uses key_dim**-0.5
        self.scale = (query_dim // num_head) ** -0.5
        self.softmax = nn.Softmax(-1)
        self.layer_scale = LayerScale2D(out_c, layer_scale_init_value)

    def forward(self, x):
        identity = x
        x = self.norm(x)
        bsz, _, h, w = tuple(x.shape)

        q = self.query_proj(x).reshape(
            [bsz, self.num_head, self.query_dim // self.num_head, h * w]
        )
        q = q.transpose([0, 1, 3, 2]) * self.scale  # [B, H, HW, K]

        if self.kv_stride > 1:
            k = self.key_proj(self.key_down_proj(x))
            v = self.value_proj(self.value_down_proj(x))
        else:
            k = self.key_proj(x)
            v = self.value_proj(x)

        k_h, k_w = tuple(k.shape[-2:])
        k = k.reshape([bsz, 1, self.kv_dim, k_h * k_w]).transpose([0, 1, 3, 2])
        v = v.reshape([bsz, 1, self.kv_dim, k_h * k_w]).transpose([0, 1, 3, 2])

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
        attn = self.softmax(attn)
        out = paddle.matmul(attn, v)  # [B, H, HW, K]
        out = out.transpose([0, 1, 3, 2]).reshape([bsz, self.query_dim, h, w])

        x = self.proj(out)
        x = self.layer_scale(x)
        if self.if_shortcut:
            x = x + identity
        return x


class MobileNetV5MultiScaleFusionAdapter(nn.Layer):

    def __init__(
        self,
        in_chs,
        out_chs,
        output_resolution=16,
        expansion_ratio=2.0,
        layer_scale_init_value=None,
        act="gelu",
    ):
        super().__init__()
        mid_chs = _make_divisible(in_chs * expansion_ratio)
        self.output_resolution = output_resolution
        self.ffn = UniversalInvertedResidualV5(
            in_c=in_chs,
            mid_c=mid_chs,
            out_c=out_chs,
            filter_size=0,
            stem_kernel_size=0,
            stride=1,
            layer_scale_init_value=layer_scale_init_value,
            act=act,
        )
        self.norm = RmsNorm2D(out_chs)

    def forward(self, inputs):
        high_h, high_w = tuple(inputs[0].shape[-2:])
        resized = []
        for feat in inputs:
            if tuple(feat.shape[-2:]) != (high_h, high_w):
                feat = F.interpolate(feat, size=[high_h, high_w], mode="nearest")
            resized.append(feat)
        x = paddle.concat(resized, axis=1)
        x = self.ffn(x)

        if high_h != self.output_resolution or high_w != self.output_resolution:
            if (
                high_h % self.output_resolution == 0
                and high_w % self.output_resolution == 0
            ):
                stride_h = high_h // self.output_resolution
                stride_w = high_w // self.output_resolution
                x = F.avg_pool2d(
                    x, kernel_size=[stride_h, stride_w], stride=[stride_h, stride_w]
                )
            else:
                x = F.interpolate(
                    x,
                    size=[self.output_resolution, self.output_resolution],
                    mode="bilinear",
                )

        return self.norm(x)


class MobileNetV5(TheseusLayer):

    def __init__(
        self,
        arch_def,
        class_num=1000,
        inplanes=64,
        drop_path_rate=0.0,
        drop_rate=0.0,
        layer_scale_init_value=1e-5,
        msfa_indices=(-2, -1),
        msfa_output_resolution=16,
        class_expand=2048,
        pad_type="",
        encoder=False,
        act="gelu",
        return_patterns=None,
        return_stages=None,
        **kwargs,
    ):
        super().__init__()
        self.class_num = class_num
        self.encoder = encoder
        self.drop_rate = drop_rate
        self.msfa_output_resolution = msfa_output_resolution

        self.conv_stem = ConvBnActV5(
            in_c=3,
            out_c=inplanes,
            filter_size=3,
            stride=2,
            padding=_get_padding(3, pad_type),
            conv_bias=True,
            act=act,
        )

        blocks = []
        stage_ends = []
        stage_out_channels = []
        all_blocks = [b for stage in arch_def for b in stage]
        block_total = len(all_blocks)

        in_c = inplanes
        block_idx = 0
        for stage in arch_def:
            for inner_idx, block_str in enumerate(stage):
                block_type, p = _parse_block(block_str)
                out_c = _make_divisible(p["c"])
                dp = drop_path_rate * block_idx / max(1, block_total)
                stride = p.get("s", 1)
                # timm EfficientNetBuilder rule: only first block in each stage keeps stride > 1
                if inner_idx >= 1:
                    stride = 1

                if block_type == "er":
                    exp_ratio = p.get("e", 4)
                    mid_c = _make_divisible(in_c * exp_ratio)
                    block = EdgeResidualV5(
                        in_c=in_c,
                        mid_c=mid_c,
                        out_c=out_c,
                        filter_size=p.get("k", 3),
                        stride=stride,
                        pad_type=pad_type,
                        act=act,
                    )
                elif block_type == "uir":
                    exp_ratio = p.get("e", 4)
                    mid_c = _make_divisible(in_c * exp_ratio)
                    block = UniversalInvertedResidualV5(
                        in_c=in_c,
                        mid_c=mid_c,
                        out_c=out_c,
                        filter_size=p.get("k", 0),
                        stem_kernel_size=p.get("a", 0),
                        stride=stride,
                        pad_type=pad_type,
                        layer_scale_init_value=layer_scale_init_value,
                        act=act,
                    )
                elif block_type == "mqa":
                    num_head = p.get("h", 8)
                    kv_dim = p.get("d", 64)
                    query_dim = _make_divisible(num_head * kv_dim)
                    block = MobileAttentionV5(
                        in_c=in_c,
                        out_c=out_c,
                        filter_size=p.get("k", 3),
                        stride=stride,
                        num_head=num_head,
                        query_dim=query_dim,
                        kv_dim=_make_divisible(kv_dim),
                        kv_stride=p.get("v", 1),
                        pad_type=pad_type,
                        drop_path_rate=dp,
                        layer_scale_init_value=layer_scale_init_value,
                    )
                else:
                    raise ValueError("Unsupported block type: {}".format(block_type))

                blocks.append(block)
                in_c = out_c
                block_idx += 1

            stage_ends.append(block_idx - 1)
            stage_out_channels.append(in_c)

        self.blocks = nn.Sequential(*blocks)
        self.stage_ends = stage_ends
        self.stage_out_channels = stage_out_channels

        feature_count = len(self.stage_out_channels)
        self.msfa_indices = [i % feature_count for i in msfa_indices]
        msfa_in_chs = sum([self.stage_out_channels[i] for i in self.msfa_indices])

        self.msfa = MobileNetV5MultiScaleFusionAdapter(
            in_chs=msfa_in_chs,
            out_chs=class_expand,
            output_resolution=msfa_output_resolution,
            layer_scale_init_value=None,
            act=act,
        )

        self.global_pool = AdaptiveAvgPool2D(1)
        self.flatten = Flatten(start_axis=1, stop_axis=-1)
        self.dropout = Dropout(drop_rate)
        self.classifier = (
            Linear(class_expand, class_num) if class_num > 0 else Identity()
        )

        stages_pattern = ["blocks[{}]".format(idx) for idx in self.stage_ends]
        super().init_res(
            stages_pattern, return_patterns=return_patterns, return_stages=return_stages
        )

    def forward_features(self, x):
        x = self.conv_stem(x)
        stage_features = []
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx in self.stage_ends:
                stage_features.append(x)

        msfa_inputs = [stage_features[i] for i in self.msfa_indices]
        return self.msfa(msfa_inputs)

    def forward_head(self, x, pre_logits=False):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        if self.encoder:
            return self.forward_features(x)
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        return
    if pretrained is True:
        if not model_url:
            raise RuntimeError(
                "No pretrained url is configured for this MobileNetV5 variant."
            )
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError("pretrained type is not available.")


def MobileNetV5_300M(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNetV5(arch_def=_ARCH_DEFS["mobilenetv5_300m"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV5_300M"], use_ssld)
    return model


def MobileNetV5_300m(pretrained=False, use_ssld=False, **kwargs):
    return MobileNetV5_300M(pretrained=pretrained, use_ssld=use_ssld, **kwargs)


def MobileNetV5_300M_enc(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNetV5(
        arch_def=_ARCH_DEFS["mobilenetv5_300m"],
        class_num=0,
        pad_type="same",
        encoder=True,
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV5_300M_enc"], use_ssld)
    return model


def MobileNetV5_300M_enc_cls(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNetV5(
        arch_def=_ARCH_DEFS["mobilenetv5_300m"], pad_type="same", **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileNetV5_300M_enc_cls"], use_ssld
    )
    return model


def MobileNetV5_base(pretrained=False, use_ssld=False, **kwargs):
    model = MobileNetV5(arch_def=_ARCH_DEFS["mobilenetv5_base"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileNetV5_base"], use_ssld)
    return model
