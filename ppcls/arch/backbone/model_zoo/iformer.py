# Copyright 2022 Garena Online Private Limited
# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Paddle implementation of Inception Transformer (iFormer).

The implementation follows the original PyTorch model while preserving its
parameter hierarchy, which makes pretrained-weight conversion straightforward.
"""

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .vision_transformer import DropPath, Identity, Mlp, ones_, trunc_normal_, zeros_
from ....utils.save_load import load_dygraph_pretrain


MODEL_URLS = {}

__all__ = [
    "iformer_small",
    "iformer_small_384",
    "iformer_base",
    "iformer_base_384",
    "iformer_large",
    "iformer_large_384",
]


class PatchEmbed(nn.Layer):
    """Downsample an NCHW feature map and return an NHWC feature map."""

    def __init__(self,
                 kernel_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.norm = nn.BatchNorm2D(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x.transpose([0, 2, 3, 1])


class FirstPatchEmbed(nn.Layer):
    """The two-convolution stem used by iFormer's first stage."""

    def __init__(self,
                 kernel_size=3,
                 stride=2,
                 padding=1,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj1 = nn.Conv2D(
            in_chans,
            embed_dim // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.norm1 = nn.BatchNorm2D(embed_dim // 2)
        self.gelu1 = nn.GELU(approximate=False)
        self.proj2 = nn.Conv2D(
            embed_dim // 2,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)
        self.norm2 = nn.BatchNorm2D(embed_dim)

    def forward(self, x):
        x = self.proj1(x)
        x = self.norm1(x)
        x = self.gelu1(x)
        x = self.proj2(x)
        x = self.norm2(x)
        return x.transpose([0, 2, 3, 1])


class HighMixer(nn.Layer):
    def __init__(self,
                 dim,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("HighMixer expects an even input dimension")

        self.cnn_in = dim // 2
        self.pool_in = dim // 2
        cnn_dim = self.cnn_in * 2
        pool_dim = self.pool_in * 2

        self.conv1 = nn.Conv2D(
            self.cnn_in, cnn_dim, kernel_size=1, bias_attr=False)
        self.proj1 = nn.Conv2D(
            cnn_dim,
            cnn_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=cnn_dim,
            bias_attr=False)
        self.mid_gelu1 = nn.GELU(approximate=False)

        # Keep the original attribute spelling for checkpoint compatibility.
        self.Maxpool = nn.MaxPool2D(
            kernel_size=kernel_size, stride=stride, padding=padding)
        self.proj2 = nn.Conv2D(
            self.pool_in, pool_dim, kernel_size=1, stride=1)
        self.mid_gelu2 = nn.GELU(approximate=False)

    def forward(self, x):
        cx = x[:, :self.cnn_in, :, :]
        cx = self.conv1(cx)
        cx = self.proj1(cx)
        cx = self.mid_gelu1(cx)

        px = x[:, self.cnn_in:, :, :]
        px = self.Maxpool(px)
        px = self.proj2(px)
        px = self.mid_gelu2(px)
        return paddle.concat([cx, px], axis=1)


class LowMixer(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 pool_size=2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("LowMixer dimension must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.dim = dim
        self.pool_size = pool_size

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.pool = (
            nn.AvgPool2D(
                kernel_size=pool_size,
                stride=pool_size,
                padding=0,
                exclusive=True)
            if pool_size > 1 else Identity())

    def forward(self, x):
        xa = self.pool(x)
        batch_size, channels, height, width = xa.shape
        num_tokens = height * width

        xa = xa.transpose([0, 2, 3, 1])
        xa = xa.reshape([batch_size, num_tokens, channels])
        qkv = self.qkv(xa).reshape([
            batch_size,
            num_tokens,
            3,
            self.num_heads,
            channels // self.num_heads,
        ])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        xa = paddle.matmul(attn, v)
        xa = xa.transpose([0, 1, 3, 2])
        xa = xa.reshape([batch_size, channels, height, width])
        if self.pool_size > 1:
            xa = F.interpolate(
                xa, scale_factor=self.pool_size, mode="nearest")
        return xa


class Mixer(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 attention_head=1,
                 pool_size=2):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("Mixer dimension must be divisible by num_heads")

        head_dim = dim // num_heads
        self.low_dim = attention_head * head_dim
        self.high_dim = dim - self.low_dim

        self.high_mixer = HighMixer(self.high_dim)
        self.low_mixer = LowMixer(
            self.low_dim,
            num_heads=attention_head,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            pool_size=pool_size)

        fused_dim = self.low_dim + self.high_dim * 2
        self.conv_fuse = nn.Conv2D(
            fused_dim,
            fused_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=fused_dim,
            bias_attr=False)
        self.proj = nn.Conv2D(fused_dim, dim, kernel_size=1)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        hx = self.high_mixer(x[:, :self.high_dim, :, :])
        lx = self.low_mixer(x[:, self.high_dim:, :, :])

        x = paddle.concat([hx, lx], axis=1)
        x = x + self.conv_fuse(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x.transpose([0, 2, 3, 1])


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 attention_head=1,
                 pool_size=2,
                 use_layer_scale=False,
                 layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = Mixer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            attention_head=attention_head,
            pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=drop)

        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            initializer = nn.initializer.Constant(layer_scale_init_value)
            self.layer_scale_1 = self.create_parameter(
                shape=[dim], default_initializer=initializer)
            self.layer_scale_2 = self.create_parameter(
                shape=[dim], default_initializer=initializer)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class InceptionTransformer(nn.Layer):
    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=None,
                 depths=None,
                 num_heads=None,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 attention_heads=None,
                 use_layer_scale=False,
                 layer_scale_init_value=1e-5):
        super().__init__()
        if not all([embed_dims, depths, num_heads, attention_heads]):
            raise ValueError("iFormer stage definitions must not be empty")
        if len(embed_dims) != 4 or len(depths) != 4 or len(num_heads) != 4:
            raise ValueError("iFormer expects exactly four stages")
        if len(attention_heads) != sum(depths):
            raise ValueError("attention_heads must contain one value per block")

        stage2_idx = depths[0]
        stage3_idx = sum(depths[:2])
        stage4_idx = sum(depths[:3])
        depth = sum(depths)
        self.num_classes = class_num

        if depth == 1:
            drop_path_rates = [drop_path_rate]
        else:
            drop_path_rates = [
                drop_path_rate * index / (depth - 1) for index in range(depth)
            ]

        self.patch_embed = FirstPatchEmbed(
            in_chans=in_chans, embed_dim=embed_dims[0])
        num_patches = img_size // 4
        self.num_patches1 = num_patches
        self.pos_embed1 = self.create_parameter(
            shape=[1, num_patches, num_patches, embed_dims[0]],
            default_initializer=nn.initializer.Constant(0.))
        self.blocks1 = nn.Sequential(*[
            Block(
                dim=embed_dims[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[index],
                attention_head=attention_heads[index],
                pool_size=2) for index in range(stage2_idx)
        ])

        self.patch_embed2 = PatchEmbed(
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1])
        num_patches //= 2
        self.num_patches2 = num_patches
        self.pos_embed2 = self.create_parameter(
            shape=[1, num_patches, num_patches, embed_dims[1]],
            default_initializer=nn.initializer.Constant(0.))
        self.blocks2 = nn.Sequential(*[
            Block(
                dim=embed_dims[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[index],
                attention_head=attention_heads[index],
                pool_size=2) for index in range(stage2_idx, stage3_idx)
        ])

        self.patch_embed3 = PatchEmbed(
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2])
        num_patches //= 2
        self.num_patches3 = num_patches
        self.pos_embed3 = self.create_parameter(
            shape=[1, num_patches, num_patches, embed_dims[2]],
            default_initializer=nn.initializer.Constant(0.))
        self.blocks3 = nn.Sequential(*[
            Block(
                dim=embed_dims[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[index],
                attention_head=attention_heads[index],
                pool_size=1,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value)
            for index in range(stage3_idx, stage4_idx)
        ])

        self.patch_embed4 = PatchEmbed(
            kernel_size=3,
            stride=2,
            padding=1,
            in_chans=embed_dims[2],
            embed_dim=embed_dims[3])
        num_patches //= 2
        self.num_patches4 = num_patches
        self.pos_embed4 = self.create_parameter(
            shape=[1, num_patches, num_patches, embed_dims[3]],
            default_initializer=nn.initializer.Constant(0.))
        self.blocks4 = nn.Sequential(*[
            Block(
                dim=embed_dims[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rates[index],
                attention_head=attention_heads[index],
                pool_size=1,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value)
            for index in range(stage4_idx, depth)
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], epsilon=1e-6)
        self.head = (nn.Linear(embed_dims[-1], class_num)
                     if class_num > 0 else Identity())
        self.init_weights()

    def init_weights(self):
        trunc_normal_(self.pos_embed1)
        trunc_normal_(self.pos_embed2)
        trunc_normal_(self.pos_embed3)
        trunc_normal_(self.pos_embed4)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(layer):
        if isinstance(layer, (nn.Linear, nn.Conv2D)):
            trunc_normal_(layer.weight)
            if layer.bias is not None:
                zeros_(layer.bias)
        elif isinstance(layer, (nn.LayerNorm, nn.BatchNorm2D)):
            if layer.bias is not None:
                zeros_(layer.bias)
            if layer.weight is not None:
                ones_(layer.weight)

    @staticmethod
    def _get_pos_embed(pos_embed, num_patches, height, width):
        if height == num_patches and width == num_patches:
            return pos_embed
        pos_embed = pos_embed.transpose([0, 3, 1, 2])
        pos_embed = F.interpolate(
            pos_embed,
            size=[height, width],
            mode="bilinear",
            align_corners=False)
        return pos_embed.transpose([0, 2, 3, 1])

    def forward_features(self, x):
        x = self.patch_embed(x)
        _, height, width, _ = x.shape
        x = x + self._get_pos_embed(
            self.pos_embed1, self.num_patches1, height, width)
        x = self.blocks1(x)

        x = self.patch_embed2(x.transpose([0, 3, 1, 2]))
        _, height, width, _ = x.shape
        x = x + self._get_pos_embed(
            self.pos_embed2, self.num_patches2, height, width)
        x = self.blocks2(x)

        x = self.patch_embed3(x.transpose([0, 3, 1, 2]))
        _, height, width, _ = x.shape
        x = x + self._get_pos_embed(
            self.pos_embed3, self.num_patches3, height, width)
        x = self.blocks3(x)

        x = self.patch_embed4(x.transpose([0, 3, 1, 2]))
        _, height, width, channels = x.shape
        x = x + self._get_pos_embed(
            self.pos_embed4, self.num_patches4, height, width)
        x = self.blocks4(x)

        x = x.reshape([x.shape[0], height * width, channels])
        x = self.norm(x)
        return paddle.mean(x, axis=1)

    def forward(self, x):
        return self.head(self.forward_features(x))


def _load_pretrained(pretrained, model, model_name):
    if pretrained is False:
        return
    if isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
        return
    if pretrained is True and model_name in MODEL_URLS:
        load_dygraph_pretrain(model, MODEL_URLS[model_name])
        return
    raise ValueError(
        "No Paddle pretrained URL is available yet; pass a local .pdparams "
        "path as pretrained instead")


def _build_iformer(model_name,
                   img_size,
                   depths,
                   embed_dims,
                   num_heads,
                   attention_heads,
                   pretrained=False,
                   class_num=1000,
                   **kwargs):
    model = InceptionTransformer(
        img_size=img_size,
        depths=depths,
        embed_dims=embed_dims,
        num_heads=num_heads,
        attention_heads=attention_heads,
        use_layer_scale=True,
        layer_scale_init_value=1e-6,
        class_num=class_num,
        **kwargs)
    _load_pretrained(pretrained, model, model_name)
    return model


def iformer_small(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_small",
        img_size=224,
        depths=[3, 3, 9, 3],
        embed_dims=[96, 192, 320, 384],
        num_heads=[3, 6, 10, 12],
        attention_heads=[1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)


def iformer_small_384(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_small_384",
        img_size=384,
        depths=[3, 3, 9, 3],
        embed_dims=[96, 192, 320, 384],
        num_heads=[3, 6, 10, 12],
        attention_heads=[1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)


def iformer_base(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_base",
        img_size=224,
        depths=[4, 6, 14, 6],
        embed_dims=[96, 192, 384, 512],
        num_heads=[3, 6, 12, 16],
        attention_heads=[1] * 4 + [3] * 6 + [8] * 7 + [10] * 7 + [15] * 6,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)


def iformer_base_384(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_base_384",
        img_size=384,
        depths=[4, 6, 14, 6],
        embed_dims=[96, 192, 384, 512],
        num_heads=[3, 6, 12, 16],
        attention_heads=[1] * 4 + [3] * 6 + [8] * 7 + [10] * 7 + [15] * 6,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)


def iformer_large(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_large",
        img_size=224,
        depths=[4, 6, 18, 8],
        embed_dims=[96, 192, 448, 640],
        num_heads=[3, 6, 14, 20],
        attention_heads=[1] * 4 + [3] * 6 + [10] * 9 + [12] * 9 + [19] * 8,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)


def iformer_large_384(pretrained=False, class_num=1000, **kwargs):
    return _build_iformer(
        "iformer_large_384",
        img_size=384,
        depths=[4, 6, 18, 8],
        embed_dims=[96, 192, 448, 640],
        num_heads=[3, 6, 14, 20],
        attention_heads=[1] * 4 + [3] * 6 + [10] * 9 + [12] * 9 + [19] * 8,
        pretrained=pretrained,
        class_num=class_num,
        **kwargs)
