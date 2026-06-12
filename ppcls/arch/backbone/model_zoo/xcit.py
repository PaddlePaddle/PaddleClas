# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/xcit.py
# reference: https://arxiv.org/abs/2106.09681

import math
from functools import partial

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    # Patch size 16, 224
    "XCiT_nano_12_p16_224": "",
    "XCiT_nano_12_p16_224_distilled": "",
    "XCiT_tiny_12_p16_224": "",
    "XCiT_tiny_12_p16_224_distilled": "",
    "XCiT_tiny_24_p16_224": "",
    "XCiT_tiny_24_p16_224_distilled": "",
    "XCiT_small_12_p16_224": "",
    "XCiT_small_12_p16_224_distilled": "",
    "XCiT_small_24_p16_224": "",
    "XCiT_small_24_p16_224_distilled": "",
    "XCiT_medium_24_p16_224": "",
    "XCiT_medium_24_p16_224_distilled": "",
    "XCiT_large_24_p16_224": "",
    "XCiT_large_24_p16_224_distilled": "",
    # Patch size 16, 384
    "XCiT_nano_12_p16_384_distilled": "",
    "XCiT_tiny_12_p16_384_distilled": "",
    "XCiT_tiny_24_p16_384_distilled": "",
    "XCiT_small_12_p16_384_distilled": "",
    "XCiT_small_24_p16_384_distilled": "",
    "XCiT_medium_24_p16_384_distilled": "",
    "XCiT_large_24_p16_384_distilled": "",
    # Patch size 8, 224
    "XCiT_nano_12_p8_224": "",
    "XCiT_nano_12_p8_224_distilled": "",
    "XCiT_tiny_12_p8_224": "",
    "XCiT_tiny_12_p8_224_distilled": "",
    "XCiT_tiny_24_p8_224": "",
    "XCiT_tiny_24_p8_224_distilled": "",
    "XCiT_small_12_p8_224": "",
    "XCiT_small_12_p8_224_distilled": "",
    "XCiT_small_24_p8_224": "",
    "XCiT_small_24_p8_224_distilled": "",
    "XCiT_medium_24_p8_224": "",
    "XCiT_medium_24_p8_224_distilled": "",
    "XCiT_large_24_p8_224": "",
    "XCiT_large_24_p8_224_distilled": "",
    # Patch size 8, 384
    "XCiT_nano_12_p8_384_distilled": "",
    "XCiT_tiny_12_p8_384_distilled": "",
    "XCiT_tiny_24_p8_384_distilled": "",
    "XCiT_small_12_p8_384_distilled": "",
    "XCiT_small_24_p8_384_distilled": "",
    "XCiT_medium_24_p8_384_distilled": "",
    "XCiT_large_24_p8_384_distilled": "",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def to_2tuple(x):
    # timm: timm.layers.to_2tuple
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.full(shape=[], fill_value=1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    # timm: timm.layers.DropPath
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Layer):
    # timm: timm.layers.Mlp
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ClassAttn(nn.Layer):
    # timm: from .cait import ClassAttn
    # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/cait.py
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        # timm: self.q(x[:, 0]).unsqueeze(1).reshape(B, 1, H, D).transpose(0, 2, 1, 3)
        q = self.q(x[:, 0]).reshape([B, self.num_heads, 1, head_dim])
        # timm: k/v: (B, N, C) -> reshape(B, N, H, D) -> permute(0, 2, 1, 3) -> (B, H, N, D)
        k = self.k(x).reshape(
            [B, N, self.num_heads, head_dim]).transpose([0, 2, 1, 3])
        v = self.v(x).reshape(
            [B, N, self.num_heads, head_dim]).transpose([0, 2, 1, 3])
        # Manual attention: q @ k^T / sqrt(d)
        scale = head_dim ** -0.5
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * scale
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x_cls = paddle.matmul(attn, v)
        x_cls = x_cls.reshape([B, 1, C])
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class PositionalEncodingFourier(nn.Layer):
    # timm: PositionalEncodingFourier
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2D(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim
        self.eps = 1e-6

    def forward(self, B, H, W):
        dtype = self.token_projection.weight.dtype
        y_embed = paddle.arange(1, H + 1).cast('float32').unsqueeze(1).tile(
            [1, 1, W])
        x_embed = paddle.arange(1, W + 1).cast('float32').tile([1, H, 1])
        y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
        dim_t = paddle.arange(self.hidden_dim).cast('float32')
        dim_t = self.temperature ** (
            2 * paddle.floor(dim_t / 2) / self.hidden_dim)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack(
            [pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()],
            axis=4).flatten(3)
        pos_y = paddle.stack(
            [pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()],
            axis=4).flatten(3)
        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        pos = self.token_projection(pos.cast(dtype))
        return pos.tile([B, 1, 1, 1])


def conv3x3(in_planes, out_planes, stride=1):
    # timm: conv3x3 (torch.nn.Conv2d + torch.nn.BatchNorm2d)
    return nn.Sequential(
        nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias_attr=False),
        nn.BatchNorm2D(out_planes),
    )


class ConvPatchEmbed(nn.Layer):
    # timm: ConvPatchEmbed
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 act_layer=nn.GELU):
        super().__init__()
        img_size = to_2tuple(img_size)
        # timm: img_size = to_2tuple(img_size) -> tuple[int, int]
        num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size == 16:
            self.proj = nn.Sequential(
                conv3x3(in_chans, embed_dim // 8, 2),
                act_layer(),
                conv3x3(embed_dim // 8, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        elif patch_size == 8:
            self.proj = nn.Sequential(
                conv3x3(in_chans, embed_dim // 4, 2),
                act_layer(),
                conv3x3(embed_dim // 4, embed_dim // 2, 2),
                act_layer(),
                conv3x3(embed_dim // 2, embed_dim, 2),
            )
        else:
            raise ValueError(
                'For convolutional projection, patch size has to be in [8, 16]'
            )

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose([0, 2, 1])
        return x, (Hp, Wp)


class LPI(nn.Layer):
    # timm: LPI (Local Patch Interaction)
    def __init__(self,
                 in_features,
                 out_features=None,
                 act_layer=nn.GELU,
                 kernel_size=3):
        super().__init__()
        out_features = out_features or in_features
        padding = kernel_size // 2
        self.conv1 = nn.Conv2D(
            in_features,
            in_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_features)
        self.act = act_layer()
        self.bn = nn.BatchNorm2D(in_features)
        self.conv2 = nn.Conv2D(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape([B, C, N]).transpose([0, 2, 1])
        return x


class ClassAttentionBlock(nn.Layer):
    # timm: ClassAttentionBlock
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eta=1.,
                 tokens_norm=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = ClassAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        if eta is not None:
            self.gamma1 = nn.Parameter(eta * paddle.ones([dim]))
            self.gamma2 = nn.Parameter(eta * paddle.ones([dim]))
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        self.tokens_norm = tokens_norm

    def forward(self, x):
        x_norm1 = self.norm1(x)
        x_attn = paddle.concat(
            [self.attn(x_norm1), x_norm1[:, 1:]], axis=1)
        x = x + self.drop_path1(self.gamma1 * x_attn)

        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x = paddle.concat(
                [self.norm2(x[:, 0:1]), x[:, 1:]], axis=1)
        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = paddle.concat([cls_token, x[:, 1:]], axis=1)
        x = x_res + self.drop_path2(x)
        return x


class XCA(nn.Layer):
    # timm: XCA (Cross-Covariance Attention)
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(paddle.ones([num_heads, 1, 1]))
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        head_dim = C // self.num_heads
        # timm: qkv.reshape(B, N, 3, H, D).permute(2, 0, 3, 4, 1) -> (3, B, H, D, N)
        qkv = self.qkv(x).reshape(
            [B, N, 3, self.num_heads, head_dim]).transpose(
            [2, 0, 3, 4, 1])
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Paper section 3.2: l2-Normalization + temperature scaling
        q = nn.functional.normalize(q, axis=-1) * self.temperature
        k = nn.functional.normalize(k, axis=-1)

        # Manual attention: q @ k^T (pre-normalized, scale=1.0)
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x = paddle.matmul(attn, v)

        x = x.transpose([0, 3, 1, 2]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class XCABlock(nn.Layer):
    # timm: XCABlock
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 proj_drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 eta=1.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm3 = norm_layer(dim)
        self.local_mp = LPI(in_features=dim, act_layer=act_layer)
        self.drop_path3 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.gamma1 = nn.Parameter(eta * paddle.ones([dim]))
        self.gamma3 = nn.Parameter(eta * paddle.ones([dim]))
        self.gamma2 = nn.Parameter(eta * paddle.ones([dim]))

    def forward(self, x, H, W):
        x = x + self.drop_path1(self.gamma1 * self.attn(self.norm1(x)))
        # NOTE official code has 3 then 2, so keeping it the same to be consistent with loaded weights
        # See https://github.com/rwightman/pytorch-image-models/pull/747#issuecomment-877795721
        x = x + self.drop_path3(self.gamma3 * self.local_mp(
            self.norm3(x), H, W))
        x = x + self.drop_path2(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class Xcit(nn.Layer):
    # timm: Xcit
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 pos_drop_rate=0.,
                 proj_drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=None,
                 norm_layer=None,
                 cls_attn_layers=2,
                 use_pos_embed=True,
                 eta=1.,
                 tokens_norm=False):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        img_size = to_2tuple(img_size)
        assert (img_size[0] % patch_size == 0) and (
            img_size[1] % patch_size == 0
        ), '`patch_size` should divide image dimensions evenly'
        norm_layer = norm_layer or partial(nn.LayerNorm, epsilon=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = class_num
        self.num_features = self.embed_dim = embed_dim
        self.global_pool = global_pool

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            act_layer=act_layer,
        )

        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        if use_pos_embed:
            self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.blocks = nn.LayerList([
            XCABlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
            ) for _ in range(depth)
        ])

        self.cls_attn_blocks = nn.LayerList([
            ClassAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_drop=drop_rate,
                attn_drop=attn_drop_rate,
                act_layer=act_layer,
                norm_layer=norm_layer,
                eta=eta,
                tokens_norm=tokens_norm,
            ) for _ in range(cls_attn_layers)
        ])

        self.norm = norm_layer(embed_dim)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(
            self.num_features,
            class_num) if class_num > 0 else nn.Identity()

        # Init weights
        trunc_normal_(self.cls_token)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            pos_encoding = self.pos_embed(B, Hp, Wp).reshape(
                [B, -1, x.shape[1]]).transpose([0, 2, 1])
            x = x + pos_encoding
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        x = paddle.concat(
            (self.cls_token.expand([B, -1, -1]), x), axis=1)

        for blk in self.cls_attn_blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits=False):
        if self.global_pool:
            x = x[:, 1:].mean(axis=1) if self.global_pool == 'avg' else x[:, 0]
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


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


# ==================== Patch size 16, image size 224 ====================


def XCiT_nano_12_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_nano_12_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p16_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p16_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p16_224"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p16_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p16_224_distilled"], use_ssld=use_ssld)
    return model


# ==================== Patch size 16, image size 384 ====================


def XCiT_nano_12_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p16_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p16_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=16, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p16_384_distilled"], use_ssld=use_ssld)
    return model


# ==================== Patch size 8, image size 224 ====================


def XCiT_nano_12_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_nano_12_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p8_224_distilled"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p8_224(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p8_224"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p8_224_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p8_224_distilled"], use_ssld=use_ssld)
    return model


# ==================== Patch size 8, image size 384 ====================


def XCiT_nano_12_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=128, depth=12, num_heads=4,
        eta=1.0, tokens_norm=False, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_nano_12_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_12_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=12, num_heads=4,
        eta=1.0, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_12_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_tiny_24_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=192, depth=24, num_heads=4,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_tiny_24_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_12_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=12, num_heads=8,
        eta=1.0, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_12_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_small_24_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=384, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_small_24_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_medium_24_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=512, depth=24, num_heads=8,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_medium_24_p8_384_distilled"], use_ssld=use_ssld)
    return model


def XCiT_large_24_p8_384_distilled(pretrained=False, use_ssld=False, **kwargs):
    model = Xcit(
        patch_size=8, embed_dim=768, depth=24, num_heads=16,
        eta=1e-5, tokens_norm=True, img_size=384, **kwargs)
    _load_pretrained(pretrained, model,
                     MODEL_URLS["XCiT_large_24_p8_384_distilled"], use_ssld=use_ssld)
    return model
