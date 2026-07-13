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

# EVA02 Vision Transformer, ported from timm.models.eva
# Paper: https://arxiv.org/abs/2303.11331
# Original: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/eva.py

import logging
import math
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

logger = logging.getLogger(__name__)
MODEL_URLS = {}

__all__ = [
    "EVA02_tiny_patch14_336",
    "EVA02_small_patch14_336",
    "EVA02_base_patch14_448",
    "EVA02_large_patch14_448",
]


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def drop_path(x, drop_prob=0.0, training=False, scale_by_keep=True):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = paddle.bernoulli(
        paddle.full(shape=shape, fill_value=keep_prob, dtype="float32")
    )
    random_tensor = random_tensor.astype(x.dtype)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor = random_tensor / keep_prob
    return x * random_tensor


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def calculate_drop_path_rates(drop_path_rate, depth):
    return paddle.linspace(0, drop_path_rate, depth).numpy().tolist()


class PatchEmbed(nn.Layer):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, bias=True
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.img_size, self.grid_size, self.num_patches = self._init_img_size(img_size)
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=bias,
        )

    def _init_img_size(self, img_size):
        img_size = to_2tuple(img_size)
        grid_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
        )
        num_patches = grid_size[0] * grid_size[1]
        return img_size, grid_size, num_patches

    def feat_ratio(self):
        return max(self.patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


# ---- RoPE helpers -----------------------------------------------------------


def rot(x):
    return paddle.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)


def rope_rotate_half(x):
    x1, x2 = paddle.chunk(x, 2, axis=-1)
    return paddle.concat([-x2, x1], axis=-1)


def apply_rot_embed_cat(x, emb, half=False):
    sin_emb, cos_emb = paddle.chunk(emb, 2, axis=-1)
    if half:
        return x * cos_emb + rope_rotate_half(x) * sin_emb
    else:
        return x * cos_emb + rot(x) * sin_emb


def pixel_freq_bands(num_bands, max_freq=224.0, linear_bands=True, dtype="float32"):
    if linear_bands:
        bands = paddle.linspace(1.0, max_freq / 2, num_bands, dtype=dtype)
    else:
        bands = 2.0 ** paddle.linspace(
            0, math.log(max_freq, 2) - 1, num_bands, dtype=dtype
        )
    return bands * math.pi


def freq_bands(num_bands, temperature=10000.0, dtype="float32"):
    exp = paddle.arange(0, num_bands, dtype="int64").astype("float32") / num_bands
    return 1.0 / (temperature**exp)


def build_fourier_pos_embed(
    feat_shape,
    bands=None,
    num_bands=64,
    max_res=224,
    temperature=10000.0,
    linear_bands=False,
    include_grid=False,
    in_pixels=True,
    ref_feat_shape=None,
    grid_offset=0.0,
    grid_indexing="ij",
    dtype="float32",
):
    if bands is None:
        if in_pixels:
            bands = pixel_freq_bands(
                num_bands, float(max_res), linear_bands=linear_bands
            )
        else:
            bands = freq_bands(num_bands, temperature=temperature)

    if grid_indexing == "xy":
        feat_shape = [feat_shape[1], feat_shape[0]] + list(feat_shape[2:])
        if ref_feat_shape is not None:
            ref_feat_shape = [ref_feat_shape[1], ref_feat_shape[0]] + list(
                ref_feat_shape[2:]
            )

    if in_pixels:
        t = [paddle.linspace(-1.0, 1.0, num=s, dtype="float32") for s in feat_shape]
    else:
        t = [
            paddle.arange(s, dtype="int64").astype("float32") + grid_offset
            for s in feat_shape
        ]

    if ref_feat_shape is not None:
        t = [x / float(f) * float(r) for x, f, r in zip(t, feat_shape, ref_feat_shape)]

    grid = paddle.stack(paddle.meshgrid(*t), axis=-1).unsqueeze(-1)
    pos = grid * bands
    pos_sin, pos_cos = paddle.sin(pos).astype(dtype), paddle.cos(pos).astype(dtype)
    return [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]


def build_rotary_pos_embed(
    feat_shape,
    bands=None,
    dim=64,
    max_res=224,
    temperature=10000.0,
    linear_bands=False,
    in_pixels=True,
    ref_feat_shape=None,
    grid_offset=0.0,
    grid_indexing="ij",
    dtype="float32",
):
    sin_emb, cos_emb = build_fourier_pos_embed(
        feat_shape,
        bands=bands,
        num_bands=dim // 4,
        max_res=max_res,
        temperature=temperature,
        linear_bands=linear_bands,
        in_pixels=in_pixels,
        ref_feat_shape=ref_feat_shape,
        grid_offset=grid_offset,
        grid_indexing=grid_indexing,
        dtype=dtype,
    )
    num_spatial_dim = 1
    for s in feat_shape:
        num_spatial_dim *= s
    sin_emb = paddle.repeat_interleave(
        sin_emb.reshape([num_spatial_dim, -1]), repeats=2, axis=-1
    )
    cos_emb = paddle.repeat_interleave(
        cos_emb.reshape([num_spatial_dim, -1]), repeats=2, axis=-1
    )
    return sin_emb, cos_emb


class RotaryEmbeddingCat(nn.Layer):
    """Pre-computed rotary position embedding with concatenated sin/cos layout."""

    def __init__(
        self,
        dim,
        max_res=224,
        temperature=10000.0,
        in_pixels=True,
        linear_bands=False,
        feat_shape=None,
        ref_feat_shape=None,
        grid_offset=0.0,
        grid_indexing="ij",
    ):
        super().__init__()
        self.dim = dim
        self.max_res = max_res
        self.temperature = temperature
        self.in_pixels = in_pixels
        self.linear_bands = linear_bands
        self.feat_shape = feat_shape
        self.ref_feat_shape = ref_feat_shape
        self.grid_offset = grid_offset
        self.grid_indexing = grid_indexing
        self._use_cached_embed = feat_shape is not None

        if self._use_cached_embed:
            num_pos = 1
            for s in feat_shape:
                num_pos *= s
            self.register_buffer(
                "pos_embed",
                paddle.empty([num_pos, dim * 2], dtype="float32"),
                persistable=False,
            )
            self.bands = None
        else:
            self.register_buffer(
                "bands", paddle.empty([dim // 4], dtype="float32"), persistable=False
            )
            self.pos_embed = None
        self.reset_parameters()

    def reset_parameters(self):
        if self._use_cached_embed:
            self.pos_embed.set_value(self._get_pos_embed_values(self.feat_shape))
        else:
            self.bands.set_value(self._compute_bands())

    def _compute_bands(self):
        if self.in_pixels:
            return pixel_freq_bands(
                self.dim // 4, float(self.max_res), linear_bands=self.linear_bands
            )
        return freq_bands(self.dim // 4, temperature=self.temperature)

    def _get_pos_embed_values(self, feat_shape, dtype="float32"):
        embeds = build_rotary_pos_embed(
            feat_shape=feat_shape,
            dim=self.dim,
            max_res=self.max_res,
            temperature=self.temperature,
            linear_bands=self.linear_bands,
            in_pixels=self.in_pixels,
            ref_feat_shape=self.ref_feat_shape,
            grid_offset=self.grid_offset,
            grid_indexing=self.grid_indexing,
            dtype=dtype,
        )
        return paddle.concat(embeds, axis=-1)

    def get_embed(self, shape=None):
        if shape is not None and self.bands is not None:
            embeds = build_rotary_pos_embed(
                shape,
                bands=self.bands,
                in_pixels=self.in_pixels,
                ref_feat_shape=self.ref_feat_shape,
                grid_offset=self.grid_offset,
                grid_indexing=self.grid_indexing,
            )
            return paddle.concat(embeds, axis=-1)
        elif self.pos_embed is not None:
            return self.pos_embed
        raise RuntimeError("get_embed() requires pre-computed pos embed or bands")

    def forward(self, x):
        pos_embed = self.get_embed(list(x.shape[2:]))
        return apply_rot_embed_cat(x, pos_embed)


def _create_rope_embed(rope_type="cat", dim=768, num_heads=12, **kwargs):
    if rope_type == "cat":
        return RotaryEmbeddingCat(dim=dim // num_heads, **kwargs)
    raise ValueError(f"Unknown RoPE type: {rope_type}")


# ---- Global pool / mask -----------------------------------------------------


def global_pool_nlc(x, pool_type="avg", num_prefix_tokens=0):
    if not pool_type:
        return x
    if pool_type == "token":
        return (
            x[:, :num_prefix_tokens].mean(axis=1) if num_prefix_tokens > 1 else x[:, 0]
        )
    x = x[:, num_prefix_tokens:]
    if pool_type == "avg":
        return x.mean(axis=1)
    if pool_type == "avgmax":
        return 0.5 * (x.amax(axis=1) + x.mean(axis=1))
    if pool_type == "max":
        return x.amax(axis=1)
    raise ValueError(f"Unknown pool type: {pool_type}")


def maybe_add_mask(attn, attn_mask):
    if attn_mask is not None:
        attn = attn + attn_mask
    return attn


# ---- MLP variants -----------------------------------------------------------


class GluMlp(nn.Layer):
    """MLP with gated linear unit. gate_last=False uses F.swiglu."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Silu,
        norm_layer=None,
        bias=True,
        drop=0.0,
        gate_last=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        assert hidden_features % 2 == 0
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.gate_last = gate_last
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features // 2)
            if norm_layer is not None
            else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features // 2, out_features, bias_attr=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        if not self.gate_last:
            x = F.swiglu(x)
        else:
            x1, x2 = paddle.chunk(x, 2, axis=-1)
            x = x1 * self.act(x2)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SwiGLU(nn.Layer):
    """SwiGLU MLP: silu(fc1_g(x)) * fc1_x(x), implemented via F.swiglu."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.Silu,
        norm_layer=None,
        bias=True,
        drop=0.0,
        align_to=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        if align_to:
            hidden_features = hidden_features + (-hidden_features % align_to)

        self.fc1_g = nn.Linear(in_features, hidden_features, bias_attr=bias[0])
        self.fc1_x = nn.Linear(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = (
            norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        )
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x_gate = self.fc1_g(x)
        x = self.fc1_x(x)
        x = F.swiglu(x_gate, x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


# ---- Attention & Block ------------------------------------------------------


class EvaAttention(nn.Layer):
    """EVA Attention with RoPE, no k-bias, and fused/unfused qkv options."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        qkv_fused=True,
        qkv_bias_separate=False,
        num_prefix_tokens=1,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        norm_layer=None,
        qk_norm=False,
        scale_norm=True,
        rotate_half=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        attn_dim = head_dim * num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.num_prefix_tokens = num_prefix_tokens
        self.qkv_bias_separate = qkv_bias_separate
        self.rotate_half = rotate_half

        if qkv_fused:
            self.qkv = nn.Linear(dim, attn_dim * 3, bias_attr=False)
            self.q_proj = self.k_proj = self.v_proj = None
            if qkv_bias:
                self.q_bias = self.create_parameter(
                    shape=[attn_dim], dtype="float32", is_bias=True
                )
                self.register_buffer(
                    "k_bias",
                    paddle.zeros([attn_dim], dtype="float32"),
                    persistable=False,
                )
                self.v_bias = self.create_parameter(
                    shape=[attn_dim], dtype="float32", is_bias=True
                )
            else:
                self.q_bias = self.k_bias = self.v_bias = None
        else:
            self.q_proj = nn.Linear(dim, attn_dim, bias_attr=qkv_bias)
            self.k_proj = nn.Linear(dim, attn_dim, bias_attr=False)
            self.v_proj = nn.Linear(dim, attn_dim, bias_attr=qkv_bias)
            self.qkv = None
            self.q_bias = self.k_bias = self.v_bias = None

        self.q_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.norm = norm_layer(attn_dim) if scale_norm else nn.Identity()
        self.proj = nn.Linear(attn_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.reset_parameters()

    def reset_parameters(self):
        if self.q_bias is not None:
            self.q_bias.set_value(paddle.zeros_like(self.q_bias))
            self.v_bias.set_value(paddle.zeros_like(self.v_bias))
        if self.k_bias is not None:
            self.k_bias.set_value(paddle.zeros_like(self.k_bias))

    def forward(self, x, rope=None, attn_mask=None):
        B, N, C = x.shape

        if self.qkv is not None:
            if self.q_bias is None:
                qkv = self.qkv(x)
            else:
                qkv_bias = paddle.concat([self.q_bias, self.k_bias, self.v_bias])
                if self.qkv_bias_separate:
                    qkv = self.qkv(x) + qkv_bias
                else:
                    qkv = F.linear(x, weight=self.qkv.weight, bias=qkv_bias)
            qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])
            q, k, v = qkv[0], qkv[1], qkv[2]
        else:
            q = (
                self.q_proj(x)
                .reshape([B, N, self.num_heads, -1])
                .transpose([0, 2, 1, 3])
            )
            k = (
                self.k_proj(x)
                .reshape([B, N, self.num_heads, -1])
                .transpose([0, 2, 1, 3])
            )
            v = (
                self.v_proj(x)
                .reshape([B, N, self.num_heads, -1])
                .transpose([0, 2, 1, 3])
            )

        q, k = self.q_norm(q), self.k_norm(k)

        if rope is not None:
            npt = self.num_prefix_tokens
            half = self.rotate_half
            q = paddle.concat(
                [
                    q[:, :, :npt, :],
                    apply_rot_embed_cat(q[:, :, npt:, :], rope, half=half),
                ],
                axis=2,
            ).astype(v.dtype)
            k = paddle.concat(
                [
                    k[:, :, :npt, :],
                    apply_rot_embed_cat(k[:, :, npt:, :], rope, half=half),
                ],
                axis=2,
            ).astype(v.dtype)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = maybe_add_mask(attn, attn_mask)
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = paddle.matmul(attn, v)

        x = x.transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.norm(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EvaBlock(nn.Layer):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qkv_fused=True,
        mlp_ratio=4.0,
        swiglu_mlp=False,
        swiglu_align_to=0,
        scale_mlp=False,
        scale_attn_inner=False,
        num_prefix_tokens=1,
        attn_type="eva",
        rotate_half=False,
        proj_drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        init_values=None,
        act_layer=nn.GELU,
        norm_layer=None,
        attn_head_dim=None,
        **kwargs,
    ):
        super().__init__()
        if norm_layer is None:
            # timm LayerNorm uses eps=1e-6; Paddle defaults to 1e-5
            norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.norm1 = norm_layer(dim)
        self.attn = EvaAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qkv_fused=qkv_fused,
            num_prefix_tokens=num_prefix_tokens,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            attn_head_dim=attn_head_dim,
            norm_layer=norm_layer,
            scale_norm=scale_attn_inner,
            rotate_half=rotate_half,
        )

        self.init_values = init_values
        self.gamma_1 = (
            self.create_parameter(shape=[dim], dtype="float32")
            if init_values is not None
            else None
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        hidden_features = int(dim * mlp_ratio)
        if scale_mlp or swiglu_align_to:
            self.mlp = SwiGLU(
                in_features=dim,
                hidden_features=hidden_features,
                norm_layer=norm_layer if scale_mlp else None,
                drop=proj_drop,
                align_to=swiglu_align_to,
            )
        else:
            self.mlp = GluMlp(
                in_features=dim,
                hidden_features=hidden_features * 2,
                norm_layer=norm_layer if scale_mlp else None,
                act_layer=nn.Silu,
                gate_last=False,
                drop=proj_drop,
            )

        self.gamma_2 = (
            self.create_parameter(shape=[dim], dtype="float32")
            if init_values is not None
            else None
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma_1 is not None:
            self.gamma_1.set_value(
                paddle.full_like(self.gamma_1, float(self.init_values))
            )
            self.gamma_2.set_value(
                paddle.full_like(self.gamma_2, float(self.init_values))
            )

    def forward(self, x, rope=None, attn_mask=None):
        if self.gamma_1 is None:
            x = x + self.drop_path1(
                self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
            )
            x = x + self.drop_path2(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path1(
                self.gamma_1 * self.attn(self.norm1(x), rope=rope, attn_mask=attn_mask)
            )
            x = x + self.drop_path2(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


# ---- Main model -------------------------------------------------------------


class Eva(TheseusLayer):
    """EVA / EVA02 Vision Transformer with absolute and rotary position embeddings.

    EVA02 variants use SwiGLU MLP, RoPE, and optional scale-norm in the MLP.
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_bias=True,
        qkv_fused=True,
        mlp_ratio=4.0,
        swiglu_mlp=False,
        swiglu_align_to=0,
        scale_mlp=False,
        scale_attn_inner=False,
        attn_type="eva",
        drop_rate=0.0,
        pos_drop_rate=0.0,
        patch_drop_rate=0.0,
        proj_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        init_values=None,
        class_token=True,
        num_reg_tokens=0,
        no_embed_class=False,
        use_abs_pos_emb=True,
        use_rot_pos_emb=False,
        rope_type="cat",
        rope_grid_offset=0.0,
        rope_grid_indexing="ij",
        rope_temperature=10000.0,
        rope_rotate_half=False,
        use_post_norm=False,
        use_pre_transformer_norm=False,
        use_post_transformer_norm=None,
        use_fc_norm=None,
        attn_pool_num_heads=None,
        attn_pool_mlp_ratio=None,
        dynamic_img_size=False,
        dynamic_img_pad=False,
        ref_feat_shape=None,
        head_init_scale=0.001,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, epsilon=1e-6)

        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.head_hidden_size = self.embed_dim = embed_dim
        self.num_prefix_tokens = (1 if class_token else 0) + num_reg_tokens
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        activate_pre_norm = use_pre_transformer_norm
        activate_fc_norm = (
            use_fc_norm if use_fc_norm is not None else (global_pool == "avg")
        )
        activate_post_norm = (
            use_post_transformer_norm
            if use_post_transformer_norm is not None
            else not activate_fc_norm
        )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            bias=not use_pre_transformer_norm,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = (
            self.create_parameter(shape=[1, 1, embed_dim], dtype="float32")
            if class_token
            else None
        )
        self.reg_token = (
            self.create_parameter(shape=[1, num_reg_tokens, embed_dim], dtype="float32")
            if num_reg_tokens
            else None
        )

        num_pos_tokens = (
            num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        )
        self.pos_embed = (
            self.create_parameter(shape=[1, num_pos_tokens, embed_dim], dtype="float32")
            if use_abs_pos_emb
            else None
        )
        self.pos_drop = nn.Dropout(p=pos_drop_rate)
        self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            rope_kwargs = dict(
                dim=embed_dim,
                num_heads=num_heads,
                feat_shape=None if dynamic_img_size else self.patch_embed.grid_size,
                temperature=rope_temperature,
                grid_indexing=rope_grid_indexing,
            )
            if rope_type == "cat":
                rope_kwargs.update(
                    dict(
                        in_pixels=False,
                        grid_offset=rope_grid_offset,
                        ref_feat_shape=ref_feat,
                    )
                )
            self.rope = _create_rope_embed(rope_type=rope_type, **rope_kwargs)
        else:
            self.rope = None

        self.norm_pre = norm_layer(embed_dim) if activate_pre_norm else nn.Identity()

        dpr = calculate_drop_path_rates(drop_path_rate, depth)
        self.blocks = nn.LayerList(
            [
                EvaBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    qkv_fused=qkv_fused,
                    mlp_ratio=mlp_ratio,
                    swiglu_mlp=swiglu_mlp,
                    swiglu_align_to=swiglu_align_to,
                    scale_mlp=scale_mlp,
                    scale_attn_inner=scale_attn_inner,
                    attn_type=attn_type,
                    rotate_half=rope_rotate_half,
                    num_prefix_tokens=self.num_prefix_tokens,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    init_values=init_values,
                )
                for i in range(depth)
            ]
        )

        r = self.patch_embed.feat_ratio()
        self.feature_info = [
            dict(module=f"blocks.{i}", num_chs=embed_dim, reduction=r)
            for i in range(depth)
        ]

        self.norm = norm_layer(embed_dim) if activate_post_norm else nn.Identity()
        self.attn_pool = None
        self.fc_norm = norm_layer(embed_dim) if activate_fc_norm else nn.Identity()
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

    def _pos_embed(self, x):
        pos_embed = self.pos_embed
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None

        to_cat = []
        if self.cls_token is not None:
            to_cat.append(paddle.expand(self.cls_token, [x.shape[0], -1, -1]))
        if self.reg_token is not None:
            to_cat.append(paddle.expand(self.reg_token, [x.shape[0], -1, -1]))

        if self.no_embed_class:
            if pos_embed is not None:
                x = x + pos_embed
            if to_cat:
                x = paddle.concat(to_cat + [x], axis=1)
        else:
            if to_cat:
                x = paddle.concat(to_cat + [x], axis=1)
            if pos_embed is not None:
                x = x + pos_embed

        x = self.pos_drop(x)
        return x, rot_pos_embed

    def forward_features(self, x):
        x = self.patch_embed(x)
        x, rot_pos_embed = self._pos_embed(x)
        x = self.norm_pre(x)
        for blk in self.blocks:
            x = blk(x, rope=rot_pos_embed)
        x = self.norm(x)
        return x

    def pool(self, x, pool_type=None):
        if self.attn_pool is not None:
            return self.attn_pool(x)
        pool_type = self.global_pool if pool_type is None else pool_type
        return global_pool_nlc(
            x, pool_type=pool_type, num_prefix_tokens=self.num_prefix_tokens
        )

    def forward_head(self, x, pre_logits=False):
        x = self.pool(x)
        x = self.fc_norm(x)
        x = self.head_drop(x)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


# ---- Weight conversion helpers ----------------------------------------------


def _compute_resize_matrix(
    old_size, new_size, interpolation="bicubic", antialias=True, dtype="float32"
):
    old_h, old_w = old_size
    new_h, new_w = new_size
    old_total, new_total = old_h * old_w, new_h * new_w
    eye = paddle.eye(old_total, dtype=dtype)
    basis = eye.reshape([old_total, 1, old_h, old_w])
    resized = F.interpolate(
        basis, size=[new_h, new_w], mode=interpolation, align_corners=False
    )
    return resized.squeeze([1]).transpose([1, 2, 0]).reshape([new_total, old_total])


def resample_patch_embed(
    patch_embed, new_size, interpolation="bicubic", antialias=True, verbose=False
):
    old_size = tuple(patch_embed.shape[-2:])
    new_size = tuple(new_size)
    if old_size == new_size:
        return patch_embed
    orig_dtype = patch_embed.dtype
    mat = _compute_resize_matrix(
        old_size, new_size, interpolation, antialias, "float32"
    )
    pinv = paddle.linalg.pinv(mat)
    c_out, c_in = patch_embed.shape[:2]
    patch_embed_f = patch_embed.reshape([c_out, c_in, -1]).astype("float32")
    return (patch_embed_f @ pinv).reshape([c_out, c_in, *new_size]).astype(orig_dtype)


def resample_abs_pos_embed(
    posemb,
    new_size,
    old_size=None,
    num_prefix_tokens=1,
    interpolation="bicubic",
    antialias=True,
    verbose=False,
):
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = (hw, hw)

    if num_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb

    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = (
        posemb.astype("float32")
        .reshape([1, old_size[0], old_size[1], -1])
        .transpose([0, 3, 1, 2])
    )
    posemb = F.interpolate(
        posemb, size=new_size, mode=interpolation, align_corners=False
    )
    posemb = (
        posemb.transpose([0, 2, 3, 1]).reshape([1, -1, embed_dim]).astype(orig_dtype)
    )

    if posemb_prefix is not None:
        posemb = paddle.concat([posemb_prefix, posemb], axis=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, interpolation="bicubic", antialias=True):
    """Remap timm EVA02 checkpoint keys to Paddle Eva model naming."""
    out_dict = {}
    state_dict = state_dict.get("model_ema", state_dict)
    state_dict = state_dict.get("model", state_dict)
    state_dict = state_dict.get("module", state_dict)
    state_dict = state_dict.get("state_dict", state_dict)

    prefix = ""
    dinov3_weights = "storage_tokens" in state_dict
    mim_weights = not dinov3_weights and "mask_token" in state_dict
    no_qkv = "blocks.0.attn.q_proj.weight" in state_dict

    for k, v in state_dict.items():
        if prefix:
            if not k.startswith(prefix):
                continue
            k = k[len(prefix) :]

        if "rope" in k and k != "rope.freqs":
            continue

        if dinov3_weights:
            if any(k.endswith(f) for f in [".periods", ".bias_mask", "mask_token"]):
                continue
            if k.startswith("local_cls_norm"):
                continue
            if k.endswith("qkv.bias"):
                q_bias_k = k.replace("qkv.bias", "q_bias")
                try:
                    model.state_dict()[q_bias_k]
                except Exception:
                    continue
                qv, kv, vv = paddle.chunk(v, 3, axis=-1)
                out_dict[q_bias_k] = qv
                out_dict[k.replace("qkv.bias", "v_bias")] = vv
                continue
            k = k.replace("ls1.gamma", "gamma_1")
            k = k.replace("ls2.gamma", "gamma_2")
            k = k.replace("storage_tokens", "reg_token")

        elif mim_weights and k in (
            "mask_token",
            "lm_head.weight",
            "lm_head.bias",
            "norm.weight",
            "norm.bias",
        ):
            if k in ("norm.weight", "norm.bias"):
                k = k.replace("norm", "fc_norm")
            else:
                continue

        if "patch_embed.proj.weight" in k:
            target_w = model.state_dict()["patch_embed.proj.weight"]
            H, W = target_w.shape[-2:]
            if v.shape[-1] != W or v.shape[-2] != H:
                v = resample_patch_embed(
                    v, (H, W), interpolation=interpolation, antialias=antialias
                )
        elif (
            k == "pos_embed" and v.shape[1] != model.state_dict()["pos_embed"].shape[1]
        ):
            npt = (
                0
                if getattr(model, "no_embed_class", False)
                else getattr(model, "num_prefix_tokens", 1)
            )
            v = resample_abs_pos_embed(
                v,
                new_size=model.patch_embed.grid_size,
                num_prefix_tokens=npt,
                interpolation=interpolation,
                antialias=antialias,
            )

        k = k.replace("mlp.ffn_ln", "mlp.norm")
        k = k.replace("attn.inner_attn_ln", "attn.norm")
        k = k.replace("mlp.w12", "mlp.fc1")
        k = k.replace("mlp.w1", "mlp.fc1_g")
        k = k.replace("mlp.w2", "mlp.fc1_x")
        k = k.replace("mlp.w3", "mlp.fc2")
        if no_qkv:
            k = k.replace("q_bias", "q_proj.bias")
            k = k.replace("v_bias", "v_proj.bias")

        out_dict[k] = v

    return out_dict


# ---- Factory functions ------------------------------------------------------


def _create_eva(variant, pretrained=None, **kwargs):
    model = Eva(**kwargs)
    if pretrained:
        if pretrained in MODEL_URLS:
            load_dygraph_pretrain(model, MODEL_URLS[pretrained])
        else:
            state_dict = paddle.load(pretrained)
            state_dict = checkpoint_filter_fn(state_dict, model)
            missing_keys, unexpected_keys = model.set_state_dict(state_dict)
            if missing_keys:
                logger.warning(
                    f"Missing keys when loading {variant}: {missing_keys[:5]}..."
                )
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys when loading {variant}: {unexpected_keys[:5]}..."
                )
    return model


def EVA02_tiny_patch14_336(pretrained=None, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    model_args.update(kwargs)
    return _create_eva("eva02_tiny_patch14_336", pretrained=pretrained, **model_args)


def EVA02_small_patch14_336(pretrained=None, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        img_size=336,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    model_args.update(kwargs)
    return _create_eva("eva02_small_patch14_336", pretrained=pretrained, **model_args)


def EVA02_base_patch14_448(pretrained=None, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        qkv_fused=False,
        mlp_ratio=4 * 2 / 3,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    model_args.update(kwargs)
    return _create_eva("eva02_base_patch14_448", pretrained=pretrained, **model_args)


def EVA02_large_patch14_448(pretrained=None, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        img_size=448,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4 * 2 / 3,
        qkv_fused=False,
        swiglu_mlp=True,
        scale_mlp=True,
        use_rot_pos_emb=True,
        ref_feat_shape=(16, 16),
    )
    model_args.update(kwargs)
    return _create_eva("eva02_large_patch14_448", pretrained=pretrained, **model_args)
