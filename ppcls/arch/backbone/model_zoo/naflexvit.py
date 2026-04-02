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
# ../pytorch-image-models/timm/models/naflexvit.py

from __future__ import absolute_import, division, print_function

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import Constant, Normal, TruncatedNormal

from ....utils.save_load import load_dygraph_pretrain
from ..base.theseus_layer import TheseusLayer
from .vision_transformer import DropPath

MODEL_URLS = {
    "naflexvit_base_patch16_gap": "",
    "naflexvit_base_patch16_par_gap": "",
    "naflexvit_base_patch16_parfac_gap": "",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)
token_normal_ = Normal(std=1e-6)


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def batch_patchify(x, patch_size, pad=False):
    patch_size = to_2tuple(patch_size)
    b, c, h, w = x.shape
    ph, pw = patch_size
    if pad and (h % ph != 0 or w % pw != 0):
        pad_h = (ph - h % ph) % ph
        pad_w = (pw - w % pw) % pw
        x = F.pad(x, [0, pad_w, 0, pad_h], data_format="NCHW")
        h, w = h + pad_h, w + pad_w

    assert h % ph == 0 and w % pw == 0, (
        "Input image size must be divisible by patch size when dynamic_img_pad=False."
    )
    nh, nw = h // ph, w // pw
    x = x.reshape([b, c, nh, ph, nw, pw]).transpose([0, 2, 4, 3, 5, 1])
    x = x.reshape([b, nh * nw, ph * pw * c])
    return x, (nh, nw)


class Identity(nn.Layer):
    def forward(self, x):
        return x


class LayerScale(nn.Layer):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = self.create_parameter(
            shape=[dim], default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x):
        return x * self.gamma


class Mlp(TheseusLayer):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(TheseusLayer):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        proj_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias_attr=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape([b, n, 3, self.num_heads, self.head_dim])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2])) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v).transpose([0, 2, 1, 3]).reshape([b, n, c])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(TheseusLayer):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        proj_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        init_values=1e-5,
        drop_path=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=norm_eps)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=norm_eps)
        self.mlp = Mlp(
            in_features=dim, hidden_features=int(dim * mlp_ratio), drop=proj_drop
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class NaFlexEmbeds(TheseusLayer):
    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        pos_embed="learned",
        pos_embed_grid_size=(16, 16),
        pos_embed_interp_mode="bicubic",
        pos_embed_ar_preserving=False,
        dynamic_img_pad=False,
        pos_drop_rate=0.0,
        reg_tokens=0,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.dynamic_img_pad = dynamic_img_pad
        self.pos_embed_interp_mode = pos_embed_interp_mode
        self.pos_embed_ar_preserving = pos_embed_ar_preserving
        self.num_reg_tokens = reg_tokens
        self.proj = nn.Linear(self.patch_size[0] * self.patch_size[1] * in_chans, embed_dim)
        self.reg_token = (
            self.create_parameter(
                shape=[1, reg_tokens, embed_dim], default_initializer=token_normal_
            )
            if reg_tokens > 0
            else None
        )
        self.pos_embed_type = pos_embed
        if pos_embed == "factorized":
            h, w = pos_embed_grid_size
            self.pos_embed_y = self.create_parameter(
                shape=[1, h, embed_dim], default_initializer=trunc_normal_
            )
            self.pos_embed_x = self.create_parameter(
                shape=[1, w, embed_dim], default_initializer=trunc_normal_
            )
            self.pos_embed = None
        elif pos_embed == "learned":
            h, w = pos_embed_grid_size
            self.pos_embed = self.create_parameter(
                shape=[1, h, w, embed_dim], default_initializer=trunc_normal_
            )
            self.pos_embed_y = None
            self.pos_embed_x = None
        else:
            self.pos_embed = None
            self.pos_embed_y = None
            self.pos_embed_x = None
        self.pos_drop = nn.Dropout(pos_drop_rate)

    def feat_ratio(self):
        return max(self.patch_size)

    def _apply_learned_pos_embed(self, x, grid_size):
        orig_h, orig_w = self.pos_embed.shape[1:3]
        if grid_size[0] == orig_h and grid_size[1] == orig_w:
            pos = self.pos_embed.reshape([1, orig_h * orig_w, -1])
        else:
            if self.pos_embed_ar_preserving:
                interp_size = [max(grid_size), max(grid_size)]
            else:
                interp_size = list(grid_size)
            use_antialias = self.pos_embed.place.is_gpu_place()
            pos = F.interpolate(
                self.pos_embed.transpose([0, 3, 1, 2]).astype("float32"),
                size=interp_size,
                mode=self.pos_embed_interp_mode,
                align_corners=False,
                antialias=use_antialias,
            )
            pos = pos[:, :, : grid_size[0], : grid_size[1]]
            pos = pos.flatten(2).transpose([0, 2, 1]).astype(x.dtype)
        return x + pos.astype(x.dtype)

    def _interp_1d(self, table, new_length):
        if table.shape[1] == new_length:
            return table
        return F.interpolate(
            table.transpose([0, 2, 1]).astype("float32"),
            size=[new_length],
            mode="linear",
            align_corners=False,
        ).transpose([0, 2, 1]).astype(table.dtype)

    def _apply_factorized_pos_embed(self, x, grid_size):
        target_h, target_w = grid_size
        if self.pos_embed_ar_preserving:
            len_y = len_x = max(target_h, target_w)
        else:
            len_y, len_x = target_h, target_w
        pos_y = self._interp_1d(self.pos_embed_y, len_y)[:, :target_h]
        pos_x = self._interp_1d(self.pos_embed_x, len_x)[:, :target_w]
        pos = (pos_y.unsqueeze(2) + pos_x.unsqueeze(1)).reshape([1, target_h * target_w, -1])
        return x + pos.astype(x.dtype)

    def forward(self, x):
        x, grid_size = batch_patchify(x, self.patch_size, pad=self.dynamic_img_pad)
        x = self.proj(x)
        if self.pos_embed_type == "learned":
            x = self._apply_learned_pos_embed(x, grid_size)
        elif self.pos_embed_type == "factorized":
            x = self._apply_factorized_pos_embed(x, grid_size)

        if self.reg_token is not None:
            reg = self.reg_token.expand([x.shape[0], -1, -1])
            x = paddle.concat([reg, x], axis=1)
        x = self.pos_drop(x)
        return x, grid_size


def global_pool_nlc(x, pool_type="avg", num_prefix_tokens=0):
    if pool_type == "token":
        return x[:, 0]
    if num_prefix_tokens:
        x = x[:, num_prefix_tokens:]
    if pool_type == "avg":
        return x.mean(axis=1)
    if pool_type == "max":
        return x.max(axis=1)
    if pool_type == "avgmax":
        return 0.5 * (x.mean(axis=1) + x.max(axis=1))
    raise ValueError("Unsupported pool_type: {}".format(pool_type))


class NaFlexVit(TheseusLayer):
    def __init__(
        self,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        in_chans=3,
        class_num=1000,
        qkv_bias=True,
        proj_bias=True,
        init_values=1e-5,
        pos_embed="learned",
        pos_embed_grid_size=(16, 16),
        pos_embed_interp_mode="bicubic",
        pos_embed_ar_preserving=False,
        dynamic_img_pad=False,
        drop_rate=0.0,
        pos_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        reg_tokens=4,
        global_pool="avg",
        final_norm=True,
        fc_norm=True,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.num_classes = class_num
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.global_pool = global_pool
        self.num_prefix_tokens = reg_tokens
        self.embeds = NaFlexEmbeds(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            pos_embed=pos_embed,
            pos_embed_grid_size=pos_embed_grid_size,
            pos_embed_interp_mode=pos_embed_interp_mode,
            pos_embed_ar_preserving=pos_embed_ar_preserving,
            dynamic_img_pad=dynamic_img_pad,
            pos_drop_rate=pos_drop_rate,
            reg_tokens=reg_tokens,
        )
        self.norm_pre = Identity()
        dpr = [x.item() for x in paddle.linspace(0.0, drop_path_rate, depth)]
        self.blocks = nn.LayerList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    init_values=init_values,
                    drop_path=dpr[i],
                    norm_eps=norm_eps,
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, epsilon=norm_eps) if final_norm and not fc_norm else Identity()
        self.fc_norm = nn.LayerNorm(embed_dim, epsilon=norm_eps) if final_norm and fc_norm else Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        return {"embeds.pos_embed", "embeds.pos_embed_y", "embeds.pos_embed_x", "embeds.reg_token"}

    def forward_features(self, x):
        x, _ = self.embeds(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = global_pool_nlc(
            x, pool_type=self.global_pool, num_prefix_tokens=self.num_prefix_tokens
        )
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        return
    if pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def naflexvit_base_patch16_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexVit(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        init_values=1e-5,
        pos_embed="learned",
        pos_embed_grid_size=(16, 16),
        reg_tokens=4,
        global_pool="avg",
        fc_norm=True,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_gap"], use_ssld)
    return model


def naflexvit_base_patch16_par_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexVit(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        init_values=1e-5,
        pos_embed="learned",
        pos_embed_grid_size=(16, 16),
        pos_embed_ar_preserving=True,
        reg_tokens=4,
        global_pool="avg",
        fc_norm=True,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["naflexvit_base_patch16_par_gap"], use_ssld)
    return model


def naflexvit_base_patch16_parfac_gap(pretrained=False, use_ssld=False, **kwargs):
    model = NaFlexVit(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        init_values=1e-5,
        pos_embed="factorized",
        pos_embed_grid_size=(16, 16),
        pos_embed_ar_preserving=True,
        reg_tokens=4,
        global_pool="avg",
        fc_norm=True,
        **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["naflexvit_base_patch16_parfac_gap"], use_ssld
    )
    return model
