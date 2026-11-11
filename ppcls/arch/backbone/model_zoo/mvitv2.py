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

import operator
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial, reduce
from typing import List, Optional, Tuple, Type, Union

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal

from .vision_transformer import zeros_, ones_, DropPath, Identity, Mlp
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MViTv2_tiny":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/mvitv2_tiny.pdparams",
    "MViTv2_small":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/mvitv2_small.pdparams",
    "MViTv2_base":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/mvitv2_base.pdparams",
    "MViTv2_large":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/mvitv2_large.pdparams",
}

__all__ = [
    'MViTv2_tiny', 'MViTv2_small', 'MViTv2_base', 'MViTv2_large',
    'MViTv2_small_cls', 'MViTv2_base_cls', 'MViTv2_large_cls', 'MViTv2_huge_cls'
]


def to_2tuple(x):
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def get_norm_layer(norm_layer='layernorm'):
    if norm_layer == 'layernorm':
        return nn.LayerNorm
    elif norm_layer == 'batchnorm':
        return nn.BatchNorm2D
    else:
        raise ValueError(f"Unsupported norm layer: {norm_layer}")


def calculate_drop_path_rates(drop_path_rate, depths, stagewise=True):
    total = sum(depths)
    flat = [drop_path_rate * i / (total - 1) for i in range(total)] if total > 1 else [0.0] * total
    if stagewise:
        dpr = []
        start = 0
        for depth in depths:
            dpr.append(flat[start:start + depth])
            start += depth
    else:
        dpr = flat
    return dpr


def trunc_normal_tf_(tensor, std=0.02):
    TruncatedNormal(std=std)(tensor)


@dataclass
class MultiScaleVitCfg:
    depths: Tuple[int, ...] = (2, 3, 16, 3)
    embed_dim: Union[int, Tuple[int, ...]] = 96
    num_heads: Union[int, Tuple[int, ...]] = 1
    mlp_ratio: float = 4.0
    pool_first: bool = False
    expand_attn: bool = True
    qkv_bias: bool = True
    use_cls_token: bool = False
    use_abs_pos: bool = False
    residual_pooling: bool = True
    mode: str = 'conv'
    kernel_qkv: Tuple[int, int] = (3, 3)
    stride_q: Optional[Tuple[Tuple[int, int]]] = ((1, 1), (2, 2), (2, 2), (2, 2))
    stride_kv: Optional[Tuple[Tuple[int, int]]] = None
    stride_kv_adaptive: Optional[Tuple[int, int]] = (4, 4)
    patch_kernel: Tuple[int, int] = (7, 7)
    patch_stride: Tuple[int, int] = (4, 4)
    patch_padding: Tuple[int, int] = (3, 3)
    rel_pos_type: str = 'spatial'
    norm_layer: Union[str, Tuple[str, str]] = 'layernorm'
    norm_eps: float = 1e-06

    def __post_init__(self):
        num_stages = len(self.depths)
        if not isinstance(self.embed_dim, (tuple, list)):
            self.embed_dim = tuple(self.embed_dim * 2 ** i for i in range(num_stages))
        assert len(self.embed_dim) == num_stages
        if not isinstance(self.num_heads, (tuple, list)):
            self.num_heads = tuple(self.num_heads * 2 ** i for i in range(num_stages))
        assert len(self.num_heads) == num_stages
        if self.stride_kv_adaptive is not None and self.stride_kv is None:
            _stride_kv = self.stride_kv_adaptive
            pool_kv_stride = []
            for i in range(num_stages):
                if min(self.stride_q[i]) > 1:
                    _stride_kv = [
                        max(_stride_kv[d] // self.stride_q[i][d], 1)
                        for d in range(len(_stride_kv))
                    ]
                pool_kv_stride.append(tuple(_stride_kv))
            self.stride_kv = tuple(pool_kv_stride)


class PatchEmbed(nn.Layer):
    def __init__(
            self,
            dim_in: int = 3,
            dim_out: int = 768,
            kernel: Tuple[int, int] = (7, 7),
            stride: Tuple[int, int] = (4, 4),
            padding: Tuple[int, int] = (3, 3),
    ):
        super().__init__()
        self.proj = nn.Conv2D(
            dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding
        )

    def forward(self, x) -> Tuple[paddle.Tensor, List[int]]:
        x = self.proj(x)
        return x.flatten(2).transpose([0, 2, 1]), x.shape[-2:]


def reshape_pre_pool(
        x, feat_size: List[int], has_cls_token: bool = True
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    H, W = feat_size
    if has_cls_token:
        cls_tok, x = x[:, :, :1, :], x[:, :, 1:, :]
    else:
        cls_tok = None
    x = x.reshape([-1, H, W, x.shape[-1]]).transpose([0, 3, 1, 2])
    return x, cls_tok


def reshape_post_pool(
        x, num_heads: int, cls_tok: Optional[paddle.Tensor] = None
) -> Tuple[paddle.Tensor, List[int]]:
    feat_size = [x.shape[2], x.shape[3]]
    L_pooled = x.shape[2] * x.shape[3]
    x = x.reshape([-1, num_heads, x.shape[1], L_pooled]).transpose([0, 1, 3, 2])
    if cls_tok is not None:
        x = paddle.concat([cls_tok, x], axis=2)
    return x, feat_size


def cal_rel_pos_type(
        attn: paddle.Tensor,
        q: paddle.Tensor,
        has_cls_token: bool,
        q_size: List[int],
        k_size: List[int],
        rel_pos_h: paddle.Tensor,
        rel_pos_w: paddle.Tensor,
):
    sp_idx = 1 if has_cls_token else 0
    q_h, q_w = q_size
    k_h, k_w = k_size
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
            paddle.arange(q_h).unsqueeze(-1) * q_h_ratio
            - paddle.arange(k_h).unsqueeze(0) * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
            paddle.arange(q_w).unsqueeze(-1) * q_w_ratio
            - paddle.arange(k_w).unsqueeze(0) * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio
    rel_h = rel_pos_h[dist_h.astype('int64')]
    rel_w = rel_pos_w[dist_w.astype('int64')]
    B, n_head, q_N, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape([B, n_head, q_h, q_w, dim])
    rel_h = paddle.einsum("byhwc,hkc->byhwk", r_q, rel_h)
    rel_w = paddle.einsum("byhwc,wkc->byhwk", r_q, rel_w)
    attn[:, :, sp_idx:, sp_idx:] = (
            attn[:, :, sp_idx:, sp_idx:].reshape([B, -1, q_h, q_w, k_h, k_w])
            + rel_h.unsqueeze(-1)
            + rel_w.unsqueeze(-2)
    ).reshape([B, -1, q_h * q_w, k_h * k_w])
    return attn


class MultiScaleAttentionPoolFirst(nn.Layer):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            feat_size: Tuple[int, int],
            num_heads: int = 8,
            qkv_bias: bool = True,
            mode: str = "conv",
            kernel_q: Tuple[int, int] = (1, 1),
            kernel_kv: Tuple[int, int] = (1, 1),
            stride_q: Tuple[int, int] = (1, 1),
            stride_kv: Tuple[int, int] = (1, 1),
            has_cls_token: bool = True,
            rel_pos_type: str = "spatial",
            residual_pooling: bool = True,
            norm_layer: Type[nn.Layer] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.q = nn.Linear(dim, dim_out, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim_out, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim_out, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2D if mode == "max" else nn.AvgPool2D
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim // num_heads if mode == "conv" else dim
            if kernel_q:
                self.pool_q = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == "spatial":
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = self.create_parameter(
                shape=[rel_sp_dim, self.head_dim],
                default_initializer=zeros_
            )
            self.rel_pos_w = self.create_parameter(
                shape=[rel_sp_dim, self.head_dim],
                default_initializer=zeros_
            )
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape
        fold_dim = 1 if self.unshared else self.num_heads
        x = x.reshape([B, N, fold_dim, -1]).transpose([0, 2, 1, 3])
        q = k = v = x
        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)
        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, v_size = reshape_post_pool(v, self.num_heads, v_tok)
        else:
            v_size = feat_size
        if self.norm_v is not None:
            v = self.norm_v(v)
        q_N = q_size[0] * q_size[1] + int(self.has_cls_token)
        q = q.transpose([0, 2, 1, 3]).reshape([B, q_N, -1])
        q = self.q(q).reshape([B, q_N, self.num_heads, -1]).transpose([0, 2, 1, 3])
        k_N = k_size[0] * k_size[1] + int(self.has_cls_token)
        k = k.transpose([0, 2, 1, 3]).reshape([B, k_N, -1])
        k = self.k(k).reshape([B, k_N, self.num_heads, -1])
        v_N = v_size[0] * v_size[1] + int(self.has_cls_token)
        v = v.transpose([0, 2, 1, 3]).reshape([B, v_N, -1])
        v = self.v(v).reshape([B, v_N, self.num_heads, -1]).transpose([0, 2, 1, 3])
        attn = q * self.scale @ k
        if self.rel_pos_type == "spatial":
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = nn.functional.softmax(attn, axis=-1)
        x = attn @ v
        if self.residual_pooling:
            x = x + q
        x = x.transpose([0, 2, 1, 3]).reshape([B, -1, self.dim_out])
        x = self.proj(x)
        return x, q_size


class MultiScaleAttention(nn.Layer):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            feat_size: Tuple[int, int],
            num_heads: int = 8,
            qkv_bias: bool = True,
            mode: str = "conv",
            kernel_q: Tuple[int, int] = (1, 1),
            kernel_kv: Tuple[int, int] = (1, 1),
            stride_q: Tuple[int, int] = (1, 1),
            stride_kv: Tuple[int, int] = (1, 1),
            has_cls_token: bool = True,
            rel_pos_type: str = "spatial",
            residual_pooling: bool = True,
            norm_layer: Type[nn.Layer] = nn.LayerNorm,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        self.head_dim = dim_out // num_heads
        self.scale = self.head_dim ** -0.5
        self.has_cls_token = has_cls_token
        padding_q = tuple([int(q // 2) for q in kernel_q])
        padding_kv = tuple([int(kv // 2) for kv in kernel_kv])

        self.qkv = nn.Linear(dim, dim_out * 3, bias_attr=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        if prod(kernel_q) == 1 and prod(stride_q) == 1:
            kernel_q = None
        if prod(kernel_kv) == 1 and prod(stride_kv) == 1:
            kernel_kv = None
        self.mode = mode
        self.unshared = mode == 'conv_unshared'
        self.norm_q, self.norm_k, self.norm_v = None, None, None
        self.pool_q, self.pool_k, self.pool_v = None, None, None
        if mode in ("avg", "max"):
            pool_op = nn.MaxPool2D if mode == "max" else nn.AvgPool2D
            if kernel_q:
                self.pool_q = pool_op(kernel_q, stride_q, padding_q)
            if kernel_kv:
                self.pool_k = pool_op(kernel_kv, stride_kv, padding_kv)
                self.pool_v = pool_op(kernel_kv, stride_kv, padding_kv)
        elif mode == "conv" or mode == "conv_unshared":
            dim_conv = dim_out // num_heads if mode == "conv" else dim_out
            if kernel_q:
                self.pool_q = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_q = norm_layer(dim_conv)
            if kernel_kv:
                self.pool_k = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_k = norm_layer(dim_conv)
                self.pool_v = nn.Conv2D(
                    dim_conv,
                    dim_conv,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=dim_conv,
                    bias_attr=False,
                )
                self.norm_v = norm_layer(dim_conv)
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

        self.rel_pos_type = rel_pos_type
        if self.rel_pos_type == "spatial":
            assert feat_size[0] == feat_size[1]
            size = feat_size[0]
            q_size = size // stride_q[1] if len(stride_q) > 0 else size
            kv_size = size // stride_kv[1] if len(stride_kv) > 0 else size
            rel_sp_dim = 2 * max(q_size, kv_size) - 1

            self.rel_pos_h = self.create_parameter(
                shape=[rel_sp_dim, self.head_dim],
                default_initializer=zeros_
            )
            self.rel_pos_w = self.create_parameter(
                shape=[rel_sp_dim, self.head_dim],
                default_initializer=zeros_
            )
            trunc_normal_tf_(self.rel_pos_h, std=0.02)
            trunc_normal_tf_(self.rel_pos_w, std=0.02)

        self.residual_pooling = residual_pooling

    def forward(self, x, feat_size: List[int]):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, -1]).transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.pool_q is not None:
            q, q_tok = reshape_pre_pool(q, feat_size, self.has_cls_token)
            q = self.pool_q(q)
            q, q_size = reshape_post_pool(q, self.num_heads, q_tok)
        else:
            q_size = feat_size
        if self.norm_q is not None:
            q = self.norm_q(q)
        if self.pool_k is not None:
            k, k_tok = reshape_pre_pool(k, feat_size, self.has_cls_token)
            k = self.pool_k(k)
            k, k_size = reshape_post_pool(k, self.num_heads, k_tok)
        else:
            k_size = feat_size
        if self.norm_k is not None:
            k = self.norm_k(k)
        if self.pool_v is not None:
            v, v_tok = reshape_pre_pool(v, feat_size, self.has_cls_token)
            v = self.pool_v(v)
            v, _ = reshape_post_pool(v, self.num_heads, v_tok)
        if self.norm_v is not None:
            v = self.norm_v(v)
        attn = q * self.scale @ k.transpose([0, 1, 3, 2])
        if self.rel_pos_type == "spatial":
            attn = cal_rel_pos_type(
                attn,
                q,
                self.has_cls_token,
                q_size,
                k_size,
                self.rel_pos_h,
                self.rel_pos_w,
            )
        attn = nn.functional.softmax(attn, axis=-1)
        x = attn @ v
        if self.residual_pooling:
            x = x + q
        x = x.transpose([0, 2, 1, 3]).reshape([B, -1, self.dim_out])
        x = self.proj(x)
        return x, q_size


class MultiScaleBlock(nn.Layer):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            num_heads: int,
            feat_size: Tuple[int, int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop_path: float = 0.0,
            norm_layer: Type[nn.Layer] = nn.LayerNorm,
            kernel_q: Tuple[int, int] = (1, 1),
            kernel_kv: Tuple[int, int] = (1, 1),
            stride_q: Tuple[int, int] = (1, 1),
            stride_kv: Tuple[int, int] = (1, 1),
            mode: str = "conv",
            has_cls_token: bool = True,
            expand_attn: bool = False,
            pool_first: bool = False,
            rel_pos_type: str = "spatial",
            residual_pooling: bool = True,
    ):
        super().__init__()
        proj_needed = dim != dim_out
        self.dim = dim
        self.dim_out = dim_out
        self.has_cls_token = has_cls_token

        self.norm1 = norm_layer(dim)

        self.shortcut_proj_attn = (
            nn.Linear(dim, dim_out) if proj_needed and expand_attn else None
        )
        if stride_q and prod(stride_q) > 1:
            kernel_skip = [(s + 1 if s > 1 else s) for s in stride_q]
            stride_skip = stride_q
            padding_skip = [int(skip // 2) for skip in kernel_skip]
            self.shortcut_pool_attn = nn.MaxPool2D(
                kernel_size=kernel_skip, stride=stride_skip, padding=padding_skip
            )
        else:
            self.shortcut_pool_attn = None
        att_dim = dim_out if expand_attn else dim
        attn_layer = MultiScaleAttentionPoolFirst if pool_first else MultiScaleAttention
        self.attn = attn_layer(
            dim,
            att_dim,
            num_heads=num_heads,
            feat_size=feat_size,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=norm_layer,
            has_cls_token=has_cls_token,
            mode=mode,
            rel_pos_type=rel_pos_type,
            residual_pooling=residual_pooling,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = norm_layer(att_dim)
        mlp_dim_out = dim_out
        self.shortcut_proj_mlp = (
            nn.Linear(dim, dim_out) if proj_needed and not expand_attn else None
        )
        self.mlp = Mlp(
            in_features=att_dim,
            hidden_features=int(att_dim * mlp_ratio),
            out_features=mlp_dim_out,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else Identity()

    def _shortcut_pool(self, x, feat_size: List[int]):
        if self.shortcut_pool_attn is None:
            return x
        if self.has_cls_token:
            cls_tok, x = x[:, :1, :], x[:, 1:, :]
        else:
            cls_tok = None
        B, L, C = x.shape
        H, W = feat_size
        x = x.reshape([B, H, W, C]).transpose([0, 3, 1, 2])
        x = self.shortcut_pool_attn(x)
        x = x.reshape([B, C, -1]).transpose([0, 2, 1])
        if cls_tok is not None:
            x = paddle.concat([cls_tok, x], axis=1)
        return x

    def forward(self, x, feat_size: List[int]):
        x_norm = self.norm1(x)
        x_shortcut = (
            x if self.shortcut_proj_attn is None else self.shortcut_proj_attn(x_norm)
        )
        x_shortcut = self._shortcut_pool(x_shortcut, feat_size)
        x, feat_size_new = self.attn(x_norm, feat_size)
        x = x_shortcut + self.drop_path1(x)
        x_norm = self.norm2(x)
        x_shortcut = (
            x if self.shortcut_proj_mlp is None else self.shortcut_proj_mlp(x_norm)
        )
        x = x_shortcut + self.drop_path2(self.mlp(x_norm))
        return x, feat_size_new


class MultiScaleVitStage(nn.Layer):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            depth: int,
            num_heads: int,
            feat_size: Tuple[int, int],
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            kernel_q: Tuple[int, int] = (1, 1),
            kernel_kv: Tuple[int, int] = (1, 1),
            stride_q: Tuple[int, int] = (1, 1),
            stride_kv: Tuple[int, int] = (1, 1),
            mode: str = "conv",
            has_cls_token: bool = True,
            expand_attn: bool = False,
            pool_first: bool = False,
            rel_pos_type: str = "spatial",
            residual_pooling: bool = True,
            norm_layer: Type[nn.Layer] = nn.LayerNorm,
            drop_path: Union[float, List[float]] = 0.0,
    ):
        super().__init__()
        self.grad_checkpointing = False
        self.blocks = nn.LayerList()
        if expand_attn:
            out_dims = (dim_out,) * depth
        else:
            out_dims = (dim,) * (depth - 1) + (dim_out,)
        for i in range(depth):
            attention_block = MultiScaleBlock(
                dim=dim,
                dim_out=out_dims[i],
                num_heads=num_heads,
                feat_size=feat_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                kernel_q=kernel_q,
                kernel_kv=kernel_kv,
                stride_q=stride_q if i == 0 else (1, 1),
                stride_kv=stride_kv,
                mode=mode,
                has_cls_token=has_cls_token,
                pool_first=pool_first,
                rel_pos_type=rel_pos_type,
                residual_pooling=residual_pooling,
                expand_attn=expand_attn,
                norm_layer=norm_layer,
                drop_path=drop_path[i]
                if isinstance(drop_path, (list, tuple))
                else drop_path,
            )
            dim = out_dims[i]
            self.blocks.append(attention_block)
            if i == 0:
                feat_size = tuple(
                    [(size // stride) for size, stride in zip(feat_size, stride_q)]
                )
        self.feat_size = feat_size

    def forward(self, x, feat_size: Tuple[int, int]):
        for blk in self.blocks:
            x, feat_size = blk(x, feat_size)
        return x, feat_size


class MultiScaleVit(nn.Layer):
    def __init__(
            self,
            cfg: MultiScaleVitCfg,
            img_size: Tuple[int, int] = (224, 224),
            in_chans: int = 3,
            global_pool: Optional[str] = None,
            num_classes: int = 1000,
            class_num: int = None,
            drop_path_rate: float = 0.0,
            drop_rate: float = 0.0,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        norm_layer = partial(get_norm_layer(cfg.norm_layer), epsilon=cfg.norm_eps)
        # 支持 class_num 参数，与 PaddleClas 框架保持一致
        if class_num is not None:
            num_classes = class_num
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        if global_pool is None:
            global_pool = "token" if cfg.use_cls_token else "avg"
        self.global_pool = global_pool
        self.depths = tuple(cfg.depths)
        self.expand_attn = cfg.expand_attn

        embed_dim = cfg.embed_dim[0]
        self.patch_embed = PatchEmbed(
            dim_in=in_chans,
            dim_out=embed_dim,
            kernel=cfg.patch_kernel,
            stride=cfg.patch_stride,
            padding=cfg.patch_padding,
        )
        patch_dims = (
            img_size[0] // cfg.patch_stride[0],
            img_size[1] // cfg.patch_stride[1],
        )
        num_patches = prod(patch_dims)
        if cfg.use_cls_token:
            self.cls_token = self.create_parameter(
                shape=[1, 1, embed_dim],
                default_initializer=zeros_
            )
            self.num_prefix_tokens = 1
            pos_embed_dim = num_patches + 1
        else:
            self.num_prefix_tokens = 0
            self.cls_token = None
            pos_embed_dim = num_patches
        if cfg.use_abs_pos:
            self.pos_embed = self.create_parameter(
                shape=[1, pos_embed_dim, embed_dim],
                default_initializer=zeros_
            )
        else:
            self.pos_embed = None
        num_stages = len(cfg.embed_dim)
        feat_size = patch_dims
        curr_stride = max(cfg.patch_stride)
        dpr = calculate_drop_path_rates(
            drop_path_rate, cfg.depths, stagewise=True
        )
        self.stages = nn.LayerList()
        self.feature_info = []
        for i in range(num_stages):
            if cfg.expand_attn:
                dim_out = cfg.embed_dim[i]
            else:
                dim_out = cfg.embed_dim[min(i + 1, num_stages - 1)]
            stage = MultiScaleVitStage(
                dim=embed_dim,
                dim_out=dim_out,
                depth=cfg.depths[i],
                num_heads=cfg.num_heads[i],
                feat_size=feat_size,
                mlp_ratio=cfg.mlp_ratio,
                qkv_bias=cfg.qkv_bias,
                mode=cfg.mode,
                pool_first=cfg.pool_first,
                expand_attn=cfg.expand_attn,
                kernel_q=cfg.kernel_qkv,
                kernel_kv=cfg.kernel_qkv,
                stride_q=cfg.stride_q[i],
                stride_kv=cfg.stride_kv[i],
                has_cls_token=cfg.use_cls_token,
                rel_pos_type=cfg.rel_pos_type,
                residual_pooling=cfg.residual_pooling,
                norm_layer=norm_layer,
                drop_path=dpr[i],
            )
            curr_stride *= max(cfg.stride_q[i])
            self.feature_info += [
                dict(module=f"block.{i}", num_chs=dim_out, reduction=curr_stride)
            ]
            embed_dim = dim_out
            feat_size = stage.feat_size
            self.stages.append(stage)
        self.num_features = self.head_hidden_size = embed_dim
        self.norm = norm_layer(embed_dim)
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("drop", nn.Dropout(self.drop_rate)),
                    (
                        "fc",
                        nn.Linear(self.num_features, num_classes)
                        if num_classes > 0
                        else Identity(),
                    ),
                ]
            )
        )
        if self.pos_embed is not None:
            trunc_normal_tf_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            trunc_normal_tf_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_tf_(m.weight, std=0.02)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def no_weight_decay(self):
        return {
            k
            for k, _ in self.named_parameters()
            if any(n in k for n in ["pos_embed", "rel_pos_h", "rel_pos_w", "cls_token"])
        }

    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    def get_classifier(self) -> nn.Layer:
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = nn.Sequential(
            OrderedDict(
                [
                    ("drop", nn.Dropout(self.drop_rate)),
                    (
                        "fc",
                        nn.Linear(self.num_features, num_classes)
                        if num_classes > 0
                        else Identity(),
                    ),
                ]
            )
        )

    def forward_features(self, x):
        x, feat_size = self.patch_embed(x)
        B, N, _ = x.shape
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat([cls_tokens, x], axis=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        for stage in self.stages:
            x, feat_size = stage(x, feat_size)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool:
            if self.global_pool == "avg":
                x = x[:, self.num_prefix_tokens:].mean(1)
            else:
                x = x[:, 0]
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


model_cfgs = dict(
    mvitv2_tiny=MultiScaleVitCfg(depths=(1, 2, 5, 2)),
    mvitv2_small=MultiScaleVitCfg(depths=(1, 2, 11, 2)),
    mvitv2_base=MultiScaleVitCfg(depths=(2, 3, 16, 3)),
    mvitv2_large=MultiScaleVitCfg(
        depths=(2, 6, 36, 4), embed_dim=144, num_heads=2, expand_attn=False
    ),
    mvitv2_small_cls=MultiScaleVitCfg(depths=(1, 2, 11, 2), use_cls_token=True),
    mvitv2_base_cls=MultiScaleVitCfg(depths=(2, 3, 16, 3), use_cls_token=True),
    mvitv2_large_cls=MultiScaleVitCfg(
        depths=(2, 6, 36, 4),
        embed_dim=144,
        num_heads=2,
        use_cls_token=True,
        expand_attn=True,
    ),
    mvitv2_huge_cls=MultiScaleVitCfg(
        depths=(4, 8, 60, 8),
        embed_dim=192,
        num_heads=3,
        use_cls_token=True,
        expand_attn=True,
    ),
)


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


def _create_mvitv2(variant, cfg_variant=None, pretrained=False, **kwargs):
    cfg = model_cfgs[variant] if not cfg_variant else model_cfgs[cfg_variant]
    model = MultiScaleVit(
        cfg=cfg,
        **kwargs,
    )
    return model


def MViTv2_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = _create_mvitv2('mvitv2_tiny', **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MViTv2_tiny"], use_ssld=use_ssld)
    return model


def MViTv2_small(pretrained=False, use_ssld=False, **kwargs):
    model = _create_mvitv2('mvitv2_small', **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MViTv2_small"], use_ssld=use_ssld)
    return model


def MViTv2_base(pretrained=False, use_ssld=False, **kwargs):
    model = _create_mvitv2('mvitv2_base', **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MViTv2_base"], use_ssld=use_ssld)
    return model


def MViTv2_large(pretrained=False, use_ssld=False, **kwargs):
    model = _create_mvitv2('mvitv2_large', **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MViTv2_large"], use_ssld=use_ssld)
    return model


def MViTv2_small_cls(pretrained=False, **kwargs):
    return _create_mvitv2('mvitv2_small_cls', **kwargs)


def MViTv2_base_cls(pretrained=False, **kwargs):
    return _create_mvitv2('mvitv2_base_cls', **kwargs)


def MViTv2_large_cls(pretrained=False, **kwargs):
    return _create_mvitv2('mvitv2_large_cls', **kwargs)


def MViTv2_huge_cls(pretrained=False, **kwargs):
    return _create_mvitv2('mvitv2_huge_cls', **kwargs)
