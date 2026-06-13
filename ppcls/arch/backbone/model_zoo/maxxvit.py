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
#
# Code was based on https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py
# reference: https://arxiv.org/abs/2204.01696 (MaxViT)
# reference: https://arxiv.org/abs/2106.04803 (CoAtNet)

"""Unified MaxxViT Paddle implementation supporting both MaxViT and CoAtNet.

Architecture differences:
- MaxViT: block_type=('M','M','M','M'), all stages use MaxxVitBlock (MbConv + PartitionAttentionCl)
- CoAtNet: block_type=('C','C','T','T'), first two stages MbConvBlock, last two TransformerBlock2d
"""

import math
from collections import OrderedDict
from functools import partial
from typing import Optional, Tuple, List

import paddle
import paddle.nn as nn

__all__ = [
    "MaxxVit",
    "create_maxvit",
    "create_coatnet",
    # MaxViT factory functions
    "MaxViT_tiny_tf_224", "MaxViT_tiny_tf_384", "MaxViT_tiny_tf_512",
    "MaxViT_small_tf_224", "MaxViT_small_tf_384", "MaxViT_small_tf_512",
    "MaxViT_base_tf_224", "MaxViT_base_tf_384", "MaxViT_base_tf_512",
    "MaxViT_large_tf_224", "MaxViT_large_tf_384", "MaxViT_large_tf_512",
    # CoAtNet factory functions
    "CoAtNet_0_rw_224", "CoAtNet_1_rw_224", "CoAtNet_2_rw_224",
    "CoAtNet_bn_0_rw_224", "CoAtNet_nano_rw_224",
    "CoAtNet_rmlp_1_rw_224", "CoAtNet_rmlp_1_rw2_224",
    "CoAtNet_rmlp_2_rw_224", "CoAtNet_rmlp_2_rw_384",
    "CoAtNet_rmlp_nano_rw_224",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_2tuple(x):
    # timm: to_2tuple
    if isinstance(x, (tuple, list)):
        return tuple(x)
    return (x, x)


def make_divisible(v, divisor=8, min_value=None):
    # timm: make_divisible
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def extend_tuple(x, n):
    # timm: extend_tuple
    if not isinstance(x, (tuple, list)):
        return (x,) * n
    x = tuple(x)
    if len(x) >= n:
        return x[:n]
    return x + (x[-1],) * (n - len(x))


def _assert(cond, msg=""):
    assert cond, msg


def _calc_drop_path_rates(drop_path_rate, depths):
    # timm: _calc_drop_path_rates -- linearly increasing drop_path across all blocks
    total = sum(depths)
    rates = [float(x) for x in paddle.linspace(0, drop_path_rate, total)]
    idx = 0
    per_stage = []
    for d in depths:
        per_stage.append(rates[idx: idx + d])
        idx += d
    return per_stage


# ---------------------------------------------------------------------------
# Basic layers
# ---------------------------------------------------------------------------

class DropPath(nn.Layer):
    """DropPath -- timm: DropPath"""
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = paddle.bernoulli(paddle.full(shape, keep_prob, dtype=x.dtype))
        return x * random_tensor / keep_prob


class Mlp(nn.Layer):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks.

    Supports tuple bias/drop for compatibility with timm's RelPosMlp.
    timm: Mlp
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class ConvMlp(nn.Layer):
    """Conv-based MLP (1x1 conv) for NCHW tensors.

    timm: ConvMlp. forward: fc1 -> norm -> act -> drop -> fc2 (single drop, before fc2).
    """
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, norm_layer=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        # timm: norm_layer if provided else Identity. CoAtNet passes None -> Identity.
        self.norm = norm_layer(hidden_features) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # timm: x = fc1 -> norm -> act -> drop -> fc2
        x = self.fc1(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class LayerNorm2d(nn.Layer):
    """LayerNorm for NCHW tensors (normalizes over channel dim).

    timm: LayerNorm2d
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.weight = paddle.create_parameter(
            shape=[num_channels], dtype="float32",
            default_initializer=nn.initializer.Constant(1.0))
        self.bias = paddle.create_parameter(
            shape=[num_channels], dtype="float32",
            default_initializer=nn.initializer.Constant(0.0))
        self.eps = eps

    def forward(self, x):
        u = x.mean(axis=1, keepdim=True)
        s = ((x - u) ** 2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.eps)
        w = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        b = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        return x * w + b


class LayerScale(nn.Layer):
    """LayerScale for channels-last tensors (B, ..., C).

    timm: LayerScale
    """
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = paddle.create_parameter(
            shape=[dim], dtype="float32",
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        return x * self.gamma


class LayerScale2d(nn.Layer):
    """LayerScale for NCHW tensors (B, C, H, W).

    timm: LayerScale2d. gamma stored as 1D [dim] (matches timm safetensors
    layout); reshaped to [1, C, 1, 1] in forward for broadcasting.
    """
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = paddle.create_parameter(
            shape=[dim], dtype="float32",
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        # timm: gamma.view(1, -1, 1, 1); return x * gamma
        gamma = self.gamma.reshape([1, -1, 1, 1])
        return x * gamma


class BatchNormAct2d(nn.BatchNorm2D):
    """BatchNorm2d + activation.

    timm: BatchNormAct2d.
    act_layer: nn.Layer class for activation (e.g. nn.Silu). If None, defaults to
    nn.GELU(approximate='tanh') for MaxViT TF; pass nn.Silu for CoAtNet.
    """
    def __init__(self, num_features, eps=1e-5, apply_act=True, act_layer=None):
        super().__init__(num_features, epsilon=eps)
        if apply_act:
            # paddle native: nn.GELU(approximate='tanh') == timm gelu_tanh
            self.act = act_layer() if act_layer is not None else nn.GELU(approximate='tanh')
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(super().forward(x))


# ---------------------------------------------------------------------------
# Conv helpers
# ---------------------------------------------------------------------------

def create_conv2d(in_chs, out_chs, kernel_size, stride=1, padding=0,
                  groups=1, bias=False):
    """Create a Conv2D layer with timm-compatible padding semantics.

    timm: create_conv2d / get_padding_value
    padding values:
      - "same": paddle native TF-style SAME padding (equivalent to timm's _SamePadConv2d)
      - "" or None: symmetric PyTorch-style padding (default in timm MaxxVitConvCfg)
      - int / tuple: explicit padding passed to nn.Conv2D
    """
    if padding == "" or padding is None:
        # timm: get_padding(kernel_size, stride) -- symmetric padding
        k = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        s = stride if isinstance(stride, int) else stride[0]
        padding = ((s - 1) + (k - 1)) // 2
    return nn.Conv2D(in_chs, out_chs, kernel_size, stride=stride,
                     padding=padding, groups=groups, bias_attr=bias)


def create_pool2d(pool_type, kernel_size, stride=None, padding=0, **kwargs):
    """timm: create_pool2d"""
    stride = stride or kernel_size
    if pool_type == "avg":
        return nn.AvgPool2D(kernel_size, stride=stride, padding=padding,
                            exclusive=False)
    elif pool_type == "max":
        return nn.MaxPool2D(kernel_size, stride=stride, padding=padding)
    elif pool_type == "avg2":
        return nn.AvgPool2D(kernel_size, stride=stride, padding=padding,
                            exclusive=False)
    elif pool_type == "max2":
        return nn.MaxPool2D(kernel_size, stride=stride, padding=padding)
    raise ValueError(f"Unknown pool type: {pool_type}")


# ---------------------------------------------------------------------------
# Relative position bias
# ---------------------------------------------------------------------------

def _generate_lookup_tensor(length):
    """One-hot lookup tensor for TF-compatible relative position bias (MaxViT).

    timm: _generate_lookup_tensor (in RelPosBiasTf).
    Returns [L, L, 2L-1] one-hot tensor where entry [i, j, j-i+L-1] = 1.
    Vectorized: replaces original Python double for-loop.
    """
    max_rel = length - 1
    vocab = 2 * max_rel + 1
    idx = paddle.arange(length).reshape([length, 1])  # i: [L, 1]
    jdx = paddle.arange(length).reshape([1, length])  # j: [1, L]
    rel = jdx - idx + max_rel  # [L, L], values in [0, 2*max_rel]
    return nn.functional.one_hot(rel, num_classes=vocab).astype("float32")


def gen_relative_position_index(q_size):
    """Generate relative position index for Swin-style relative position bias.

    timm: gen_relative_position_index (in pos_embed_rel.py)
    Returns: [Wh*Ww, Wh*Ww] tensor of indices into the bias table.
    """
    coords_h = paddle.arange(q_size[0])
    coords_w = paddle.arange(q_size[1])
    grid_h, grid_w = paddle.meshgrid(coords_h, coords_w)
    coords = paddle.stack([grid_h.flatten(), grid_w.flatten()])  # [2, Wh*Ww]

    relative_coords = coords.unsqueeze(2) - coords.unsqueeze(1)  # [2, N, N]
    relative_coords = relative_coords.transpose([1, 2, 0])  # [N, N, 2]
    relative_coords = relative_coords + paddle.to_tensor(
        [q_size[0] - 1, q_size[1] - 1]).reshape([1, 1, 2])
    scale = paddle.to_tensor([2 * q_size[1] - 1, 1]).reshape([1, 1, 2])
    relative_coords = relative_coords * scale
    return relative_coords.sum(-1)  # [N, N]


def gen_relative_log_coords(win_size, mode='cr'):
    """Generate log-coordinate table for MLP-based relative position (RelPosMlp).

    timm: gen_relative_log_coords (in pos_embed_rel.py)
    Returns: [2*Wh-1, 2*Ww-1, 2] tensor.
    """
    rel_coords_h = paddle.arange(-(win_size[0] - 1), win_size[0]).astype('float32')
    rel_coords_w = paddle.arange(-(win_size[1] - 1), win_size[1]).astype('float32')
    grid_h, grid_w = paddle.meshgrid(rel_coords_h, rel_coords_w)
    table = paddle.stack([grid_h, grid_w])  # [2, 2Wh-1, 2Ww-1]
    table = table.transpose([1, 2, 0])  # [2Wh-1, 2Ww-1, 2]

    if mode == 'swin':
        table = table / paddle.to_tensor(
            [win_size[0] - 1, win_size[1] - 1]).reshape([1, 1, 2])
        table = table * 8
        table = paddle.sign(table) * paddle.log2(1.0 + paddle.abs(table)) / math.log2(8)
    else:  # 'cr'
        table = paddle.sign(table) * paddle.log(1.0 + paddle.abs(table))

    return table


class RelPosBiasTf(nn.Layer):
    """TF-compatible relative position bias (for maxvit_tf models).

    timm: RelPosBiasTf. Uses einsum-based lookup matching TensorFlow MaxViT.
    """
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        ws = window_size
        vocab_h = 2 * ws[0] - 1
        vocab_w = 2 * ws[1] - 1
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[num_heads, vocab_h, vocab_w], dtype="float32",
            default_initializer=nn.initializer.Normal(std=0.02))
        self.register_buffer("height_lookup", _generate_lookup_tensor(ws[0]))
        self.register_buffer("width_lookup", _generate_lookup_tensor(ws[1]))

    def get_bias(self):
        # timm: einsum-based reindex
        t = self.relative_position_bias_table
        hl = self.height_lookup
        wl = self.width_lookup
        reindexed = paddle.einsum("nhw,ixh->nixw", t, hl)
        reindexed = paddle.einsum("nixw,jyw->nijxy", reindexed, wl)
        area = self.window_size[0] * self.window_size[1]
        return reindexed.reshape([self.num_heads, area, area])

    def forward(self, attn):
        bias = self.get_bias().unsqueeze(0)
        return attn + bias


class RelPosBias(nn.Layer):
    """Swin-style relative position bias (for CoAtNet non-rmlp models).

    timm: RelPosBias. Uses a learnable bias table indexed by relative position.
    """
    def __init__(self, window_size, num_heads, prefix_tokens=0):
        super().__init__()
        _assert(prefix_tokens <= 1)
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.prefix_tokens = prefix_tokens
        self.bias_shape = (self.window_area + prefix_tokens,) * 2 + (num_heads,)

        num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3 * prefix_tokens
        self.relative_position_bias_table = paddle.create_parameter(
            shape=[num_relative_distance, num_heads], dtype="float32",
            default_initializer=nn.initializer.Normal(std=0.02))
        index_size = (self.window_area + prefix_tokens) ** 2
        self.register_buffer(
            "relative_position_index",
            paddle.zeros([index_size], dtype='int64'), persistent=False)
        self._init_index()

    def _init_index(self):
        idx = gen_relative_position_index(self.window_size).reshape([-1])
        self.relative_position_index.copy_(idx)

    def get_bias(self):
        # timm: self.relative_position_bias_table[self.relative_position_index]
        relative_position_bias = paddle.gather(
            self.relative_position_bias_table, self.relative_position_index, axis=0)
        relative_position_bias = relative_position_bias.reshape(self.bias_shape)
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        return relative_position_bias.unsqueeze(0)

    def forward(self, attn, shared_rel_pos=None):
        return attn + self.get_bias()


class RelPosMlp(nn.Layer):
    """Log-coordinate MLP relative position bias (for CoAtNet rmlp models).

    timm: RelPosMlp. Based on Swin-V2 ideas. Uses an MLP on log-coordinates.
    """
    def __init__(self, window_size, num_heads=8, hidden_dim=128,
                 prefix_tokens=0, mode='cr'):
        super().__init__()
        self.window_size = window_size
        self.window_area = window_size[0] * window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        self.mode = mode

        # timm: Mlp(in=2, hidden=hidden_dim, out=num_heads, act=ReLU, bias=True, drop=(0.125, 0.))
        self.mlp = Mlp(
            2,
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            bias=True,
            drop=(0.125, 0.))

        index_size = self.window_area ** 2
        self.register_buffer(
            "relative_position_index",
            paddle.zeros([index_size], dtype='int64'), persistent=False)
        rel_coords_shape = (2 * window_size[0] - 1, 2 * window_size[1] - 1, 2)
        self.register_buffer(
            "rel_coords_log",
            paddle.zeros(rel_coords_shape), persistent=False)
        self._init_buffers()

    def _init_buffers(self):
        idx = gen_relative_position_index(self.window_size).reshape([-1])
        self.relative_position_index.copy_(idx)
        self.rel_coords_log.copy_(
            gen_relative_log_coords(self.window_size, mode=self.mode))

    def get_bias(self):
        # timm: mlp(rel_coords_log).reshape(-1, num_heads)[rel_pos_idx]
        relative_position_bias = self.mlp(self.rel_coords_log)
        relative_position_bias = paddle.gather(
            relative_position_bias.reshape([-1, self.num_heads]),
            self.relative_position_index, axis=0)
        relative_position_bias = relative_position_bias.reshape(self.bias_shape)
        relative_position_bias = relative_position_bias.transpose([2, 0, 1])
        return relative_position_bias.unsqueeze(0)

    def forward(self, attn, shared_rel_pos=None):
        return attn + self.get_bias()


# ---------------------------------------------------------------------------
# Attention mechanisms
# ---------------------------------------------------------------------------

class AttentionCl(nn.Layer):
    """Channels-last multi-head attention (B, ..., C).

    timm: AttentionCl. Used by MaxViT's PartitionAttentionCl.
    """
    def __init__(self, dim, dim_out=None, dim_head=32, bias=True,
                 expand_first=True, head_first=True,
                 rel_pos_cls=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim_attn * 3, bias_attr=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_attn, dim_out, bias_attr=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # timm: head_first -> (B, num_heads, dim_head*3, -1) chunk(3, dim=2)
        #       else       -> (B, -1, 3, num_heads, dim_head) transpose(1,3) unbind(2)
        B = x.shape[0]
        # paddle: x.shape 返回 list, 需显式 list 化后再拼接; timm 原版 restore_shape = x.shape[:-1] (tuple)
        restore_shape = list(x.shape[:-1])

        if self.head_first:
            qkv = self.qkv(x).reshape(
                [B, -1, self.num_heads, self.dim_head * 3]
            ).transpose([0, 2, 1, 3])
            q, k, v = qkv.chunk(3, axis=-1)
        else:
            qkv = self.qkv(x).reshape(
                [B, -1, 3, self.num_heads, self.dim_head]
            ).transpose([0, 3, 2, 1, 4])
            q = qkv[:, :, 0, :, :]
            k = qkv[:, :, 1, :, :]
            v = qkv[:, :, 2, :, :]

        q = q * self.scale
        attn = q.matmul(k.transpose([0, 1, 3, 2]))

        if self.rel_pos is not None:
            attn = self.rel_pos(attn)

        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = attn.matmul(v)
        x = x.transpose([0, 2, 1, 3]).reshape(restore_shape + [-1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Attention2d(nn.Layer):
    """Multi-head attention for 2D NCHW tensors.

    timm: Attention2d. Used by CoAtNet's TransformerBlock2d.
    Q/K/V are computed via 1x1 Conv2D.
    """
    def __init__(self, dim, dim_out=None, dim_head=32, bias=True,
                 expand_first=True, head_first=True,
                 rel_pos_cls=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        dim_out = dim_out or dim
        dim_attn = dim_out if expand_first else dim
        self.num_heads = dim_attn // dim_head
        self.dim_head = dim_head
        self.head_first = head_first
        self.scale = dim_head ** -0.5

        self.qkv = nn.Conv2D(dim, dim_attn * 3, 1, bias_attr=bias)
        self.rel_pos = rel_pos_cls(num_heads=self.num_heads) if rel_pos_cls else None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2D(dim_attn, dim_out, 1, bias_attr=bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, shared_rel_pos=None):
        # timm:
        #   head_first -> qkv(x).view(B, num_heads, dim_head*3, -1).chunk(3, dim=2)
        #   else       -> qkv(x).reshape(B, 3, num_heads, dim_head, -1).unbind(1)
        B, C, H, W = x.shape
        qkv = self.qkv(x)  # [B, dim_attn*3, H, W]

        if self.head_first:
            # [B, dim_attn*3, H, W] -> [B, num_heads, dim_head*3, H*W]
            qkv = qkv.reshape([B, self.num_heads, self.dim_head * 3, H * W])
            q, k, v = qkv.chunk(3, axis=2)  # each [B, num_heads, dim_head, H*W]
        else:
            # [B, dim_attn*3, H, W] -> [B, 3, num_heads, dim_head, H*W]
            qkv = qkv.reshape([B, 3, self.num_heads, self.dim_head, H * W])
            q = qkv[:, 0]  # [B, num_heads, dim_head, H*W]
            k = qkv[:, 1]
            v = qkv[:, 2]

        # q, k, v: [B, num_heads, dim_head, H*W]
        # timm: q * scale, attn = q.transpose(-2,-1) @ k  ->  [B, h, HW, HW]
        q = q * self.scale
        attn = q.transpose([0, 1, 3, 2]) @ k

        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        elif shared_rel_pos is not None:
            attn = attn + shared_rel_pos

        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        # timm: x = (v @ attn.transpose(-2,-1)).view(B, -1, H, W)
        x = v @ attn.transpose([0, 1, 3, 2])
        x = x.reshape([B, -1, H, W])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------

class Downsample2d(nn.Layer):
    """2D spatial downsampling via pooling + optional 1x1 channel expansion.

    timm: Downsample2d. pool_type maps to:
      'max'  -> MaxPool2d(3, stride=2, padding=1)
      'max2' -> MaxPool2d(2, stride=2, padding=0)
      'avg'  -> AvgPool2d(3, stride=2, padding=1, count_include_pad=False)
      'avg2' -> AvgPool2d(2, stride=2, padding=0)
    """
    def __init__(self, dim, dim_out, pool_type="avg2", bias=True):
        super().__init__()
        if pool_type == "max":
            self.pool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        elif pool_type == "max2":
            self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        elif pool_type == "avg":
            # Paddle exclusive=True (default) == PyTorch count_include_pad=False
            self.pool = nn.AvgPool2D(kernel_size=3, stride=2, padding=1, exclusive=True)
        else:  # 'avg2'
            self.pool = nn.AvgPool2D(kernel_size=2, stride=2, padding=0, exclusive=True)
        if dim != dim_out:
            self.expand = nn.Conv2D(dim, dim_out, 1, bias_attr=bias)
        else:
            self.expand = nn.Identity()

    def forward(self, x):
        x = self.pool(x)
        x = self.expand(x)
        return x


# ---------------------------------------------------------------------------
# Squeeze-and-Excitation
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Layer):
    """SE channel attention for MbConvBlock.

    timm: SqueezeExcite. act_layer: activation for the squeeze FC (default SiLU).
    """
    def __init__(self, channels, rd_channels, act_layer=nn.Silu):
        super().__init__()
        self.fc1 = nn.Conv2D(channels, rd_channels, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(rd_channels, channels, 1)

    def forward(self, x):
        x_se = x.mean(axis=[-2, -1], keepdim=True)
        x_se = self.act(self.fc1(x_se))
        x_se = paddle.nn.functional.sigmoid(self.fc2(x_se))
        return x * x_se


# ---------------------------------------------------------------------------
# MbConvBlock (supports both MaxViT TF and CoAtNet configs)
# ---------------------------------------------------------------------------

class MbConvBlock(nn.Layer):
    """Pre-Norm MBConv block.

    timm: MbConvBlock. Supports MaxViT TF config (gelu_tanh, stride_mode='dw',
    expand_output=True) and CoAtNet config (silu, stride_mode varies,
    expand_output=False).
    """
    def __init__(self, in_chs, out_chs, stride=1, expand_ratio=4.0,
                 kernel_size=3, group_size=1, output_bias=True,
                 padding="same", norm_eps=1e-3, attn_ratio=0.25,
                 pool_type="avg2", downsample_pool_type=None,
                 drop_path=0.0,
                 stride_mode="dw", expand_output=True,
                 pre_norm_act=False, attn_early=False,
                 attn_act_layer=None, norm_act_layer=None):
        super().__init__()
        # timm: downsample_pool_type defaults to 'avg2' (independent of pool_type);
        # __post_init__ sets it to pool_type only when explicitly None.
        if downsample_pool_type is None:
            downsample_pool_type = "avg2"
        mid_chs = make_divisible(
            (out_chs if expand_output else in_chs) * expand_ratio)
        groups = mid_chs if group_size == 1 else mid_chs // group_size

        if stride == 2:
            self.shortcut = Downsample2d(in_chs, out_chs, pool_type=pool_type,
                                         bias=output_bias)
        else:
            self.shortcut = nn.Identity()

        # stride_mode: 'dw' (stride on depthwise), 'pool' (stride via pool)
        stride_pool, stride_1, stride_2 = 1, 1, 1
        if stride_mode == "pool":
            stride_pool = stride
        else:  # 'dw'
            stride_2 = stride

        self.pre_norm = BatchNormAct2d(
            in_chs, eps=norm_eps, apply_act=pre_norm_act, act_layer=norm_act_layer)
        if stride_pool > 1:
            # timm: down uses downsample_pool_type (NOT pool_type)
            self.down = Downsample2d(in_chs, in_chs, pool_type=downsample_pool_type)
        else:
            self.down = nn.Identity()

        self.conv1_1x1 = create_conv2d(in_chs, mid_chs, 1, stride=stride_1, bias=False)
        self.norm1 = BatchNormAct2d(mid_chs, eps=norm_eps, apply_act=True, act_layer=norm_act_layer)

        self.conv2_kxk = create_conv2d(mid_chs, mid_chs, kernel_size,
                                        stride=stride_2, groups=groups, padding=padding,
                                        bias=False)

        # SE placement: attn_early -> SE before norm2, else after
        se_act = attn_act_layer if attn_act_layer is not None else nn.Silu
        rd_channels = int(attn_ratio * (out_chs if expand_output else mid_chs))
        if attn_early:
            self.se_early = SqueezeExcitation(mid_chs, rd_channels, act_layer=se_act)
            self.norm2 = BatchNormAct2d(mid_chs, eps=norm_eps, apply_act=True, act_layer=norm_act_layer)
            self.se = None
        else:
            self.se_early = None
            self.norm2 = BatchNormAct2d(mid_chs, eps=norm_eps, apply_act=True, act_layer=norm_act_layer)
            self.se = SqueezeExcitation(mid_chs, rd_channels, act_layer=se_act)

        self.conv3_1x1 = create_conv2d(mid_chs, out_chs, 1, bias=output_bias)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # timm forward order
        shortcut = self.shortcut(x)
        x = self.pre_norm(x)
        x = self.down(x)
        x = self.conv1_1x1(x)
        x = self.norm1(x)
        x = self.conv2_kxk(x)
        if self.se_early is not None:
            x = self.se_early(x)
        x = self.norm2(x)
        if self.se is not None:
            x = self.se(x)
        x = self.conv3_1x1(x)
        x = self.drop_path(x)
        return x + shortcut


# ---------------------------------------------------------------------------
# TransformerBlock2d (NCHW, for CoAtNet transformer stages)
# ---------------------------------------------------------------------------

class TransformerBlock2d(nn.Layer):
    """Transformer block with 2D downsampling (NCHW tensor layout).

    timm: TransformerBlock2d. Used by CoAtNet for the 'T' stages.
    Uses Attention2d + ConvMlp.
    """
    def __init__(self, dim, dim_out, stride=1,
                 rel_pos_cls=None, dim_head=32, expand_first=True,
                 head_first=True, attn_bias=True,
                 norm_layer='layernorm2d', norm_eps=1e-6,
                 attn_drop=0.0, proj_drop=0.0,
                 shortcut_bias=True, pool_type='avg2',
                 mlp_ratio=4.0, act_layer=None,
                 init_values=None, drop_path=0.0):
        super().__init__()
        if norm_layer == 'batchnorm2d':
            norm = lambda c: nn.BatchNorm2D(c, epsilon=norm_eps)
        else:  # 'layernorm2d'
            norm = lambda c: LayerNorm2d(c, eps=norm_eps)

        if stride == 2:
            self.shortcut = Downsample2d(dim, dim_out, pool_type=pool_type, bias=shortcut_bias)
            self.norm1 = nn.Sequential(OrderedDict([
                ('norm', norm(dim)),
                ('down', Downsample2d(dim, dim, pool_type=pool_type)),
            ]))
        else:
            _assert(dim == dim_out, "dim must equal dim_out when stride=1")
            self.shortcut = nn.Identity()
            self.norm1 = norm(dim)

        self.attn = Attention2d(
            dim, dim_out, dim_head=dim_head, expand_first=expand_first,
            bias=attn_bias, head_first=head_first,
            rel_pos_cls=rel_pos_cls, attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale2d(dim_out, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm(dim_out)
        self.mlp = ConvMlp(
            in_features=dim_out,
            hidden_features=int(dim_out * mlp_ratio),
            act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale2d(dim_out, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, shared_rel_pos=None):
        # timm: x = shortcut(x) + drop_path1(ls1(attn(norm1(x))))
        #       x = x + drop_path2(ls2(mlp(norm2(x))))
        x = self.shortcut(x) + self.drop_path1(self.ls1(
            self.attn(self.norm1(x), shared_rel_pos=shared_rel_pos)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# Partition operations (channels-last: B, H, W, C) -- for MaxViT
# ---------------------------------------------------------------------------

def window_partition(x, window_size):
    # timm: window_partition
    B, H, W, C = x.shape
    x = x.reshape([B, H // window_size[0], window_size[0],
                   W // window_size[1], window_size[1], C])
    return x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size[0], window_size[1], C])


def window_reverse(windows, window_size, img_size):
    # timm: window_reverse
    H, W = img_size
    C = windows.shape[-1]
    x = windows.reshape([-1, H // window_size[0], W // window_size[1],
                         window_size[0], window_size[1], C])
    return x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])


def grid_partition(x, grid_size):
    # timm: grid_partition
    B, H, W, C = x.shape
    x = x.reshape([B, grid_size[0], H // grid_size[0],
                   grid_size[1], W // grid_size[1], C])
    return x.transpose([0, 2, 4, 1, 3, 5]).reshape(
        [-1, grid_size[0], grid_size[1], C])


def grid_reverse(windows, grid_size, img_size):
    # timm: grid_reverse
    H, W = img_size
    C = windows.shape[-1]
    x = windows.reshape([-1, H // grid_size[0], W // grid_size[1],
                         grid_size[0], grid_size[1], C])
    return x.transpose([0, 3, 1, 4, 2, 5]).reshape([-1, H, W, C])


# ---------------------------------------------------------------------------
# PartitionAttentionCl (channels-last, for MaxViT)
# ---------------------------------------------------------------------------

class PartitionAttentionCl(nn.Layer):
    """Grid or Block partition + Attn + FFN (channels-last layout).

    timm: PartitionAttentionCl. Used by MaxViT's MaxxVitBlock.
    """
    def __init__(self, dim, partition_type="block", window_size=None,
                 grid_size=None, dim_head=32, head_first=True,
                 act_layer=None, norm_eps=1e-5, attn_drop=0.0,
                 proj_drop=0.0, mlp_ratio=4.0, drop_path=0.0,
                 init_values=None):
        super().__init__()
        self.partition_block = partition_type == "block"
        self.partition_size = to_2tuple(
            window_size if self.partition_block else grid_size)

        rel_pos_cls = partial(RelPosBiasTf, window_size=self.partition_size)

        self.norm1 = nn.LayerNorm(dim, epsilon=norm_eps)
        self.attn = AttentionCl(
            dim, dim, dim_head=dim_head, bias=True,
            head_first=head_first, rel_pos_cls=rel_pos_cls,
            attn_drop=attn_drop, proj_drop=proj_drop)
        self.ls1 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim, epsilon=norm_eps)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), act_layer=act_layer, drop=proj_drop)
        self.ls2 = LayerScale(dim, init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def _partition_attn(self, x):
        img_size = x.shape[1:3]
        if self.partition_block:
            partitioned = window_partition(x, self.partition_size)
        else:
            partitioned = grid_partition(x, self.partition_size)
        partitioned = self.attn(partitioned)
        if self.partition_block:
            return window_reverse(partitioned, self.partition_size, img_size)
        return grid_reverse(partitioned, self.partition_size, img_size)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self._partition_attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# ---------------------------------------------------------------------------
# MaxxVitBlock (MbConv + PartitionAttentionCl block+grid, for MaxViT)
# ---------------------------------------------------------------------------

class MaxxVitBlock(nn.Layer):
    """MaxViT block: MbConv + window partition attn + grid partition attn.

    timm: MaxxVitBlock
    """
    def __init__(self, dim, dim_out, stride=1, window_size=None,
                 grid_size=None, dim_head=32, head_first=True,
                 act_layer=None, norm_eps=1e-5, attn_drop=0.0,
                 proj_drop=0.0, conv_norm_eps=1e-3,
                 conv_expand_ratio=4.0, conv_kernel_size=3,
                 conv_group_size=1, conv_output_bias=True,
                 conv_padding="same", conv_attn_ratio=0.25,
                 conv_pool_type="avg2", conv_stride_mode="dw",
                 conv_expand_output=True, conv_pre_norm_act=False,
                 conv_attn_early=False, conv_attn_act_layer=None,
                 conv_norm_act_layer=None,
                 mlp_ratio=4.0, drop_path=0.0, init_values=None):
        super().__init__()
        self.conv = MbConvBlock(
            in_chs=dim, out_chs=dim_out, stride=stride,
            expand_ratio=conv_expand_ratio, kernel_size=conv_kernel_size,
            group_size=conv_group_size, output_bias=conv_output_bias,
            padding=conv_padding, norm_eps=conv_norm_eps,
            attn_ratio=conv_attn_ratio, pool_type=conv_pool_type,
            drop_path=drop_path, stride_mode=conv_stride_mode,
            expand_output=conv_expand_output,
            pre_norm_act=conv_pre_norm_act,
            attn_early=conv_attn_early,
            attn_act_layer=conv_attn_act_layer,
            norm_act_layer=conv_norm_act_layer)

        attn_kwargs = dict(
            dim=dim_out, window_size=window_size, grid_size=grid_size,
            dim_head=dim_head, head_first=head_first,
            act_layer=act_layer, norm_eps=norm_eps,
            attn_drop=attn_drop, proj_drop=proj_drop,
            mlp_ratio=mlp_ratio, drop_path=drop_path,
            init_values=init_values)

        self.attn_block = PartitionAttentionCl(partition_type="block", **attn_kwargs)
        self.attn_grid = PartitionAttentionCl(partition_type="grid", **attn_kwargs)

    def forward(self, x):
        # timm: conv (NCHW) -> NHWC -> attn_block -> attn_grid -> NCHW
        x = self.conv(x)
        x = x.transpose([0, 2, 3, 1])  # NCHW -> NHWC
        x = self.attn_block(x)
        x = self.attn_grid(x)
        x = x.transpose([0, 3, 1, 2])  # NHWC -> NCHW
        return x


# ---------------------------------------------------------------------------
# MaxxVitStage (supports 'C', 'T', 'M' block types)
# ---------------------------------------------------------------------------

class MaxxVitStage(nn.Layer):
    """Stage of blocks.

    timm: MaxxVitStage. Supports block_type='C' (MbConv),
    'T' (Transformer2d), 'M' (MaxxVit).
    """
    def __init__(self, in_chs, out_chs, depth, stride=2,
                 block_type='M', feat_size=None, window_size=None, grid_size=None,
                 **kwargs):
        super().__init__()
        blocks = []
        for i in range(depth):
            block_stride = stride if i == 0 else 1
            drop_path = kwargs.get("drop_path_rates", [0.0] * depth)[i]
            dim_in = in_chs if i == 0 else out_chs

            if block_type == 'C':
                blocks.append(MbConvBlock(
                    in_chs=dim_in, out_chs=out_chs, stride=block_stride,
                    expand_ratio=kwargs.get("conv_expand_ratio", 4.0),
                    kernel_size=kwargs.get("conv_kernel_size", 3),
                    group_size=kwargs.get("conv_group_size", 1),
                    output_bias=kwargs.get("conv_output_bias", True),
                    padding=kwargs.get("conv_padding", "same"),
                    norm_eps=kwargs.get("conv_norm_eps", 1e-3),
                    attn_ratio=kwargs.get("conv_attn_ratio", 0.25),
                    pool_type=kwargs.get("conv_pool_type", "avg2"),
                    drop_path=drop_path,
                    stride_mode=kwargs.get("conv_stride_mode", "dw"),
                    expand_output=kwargs.get("conv_expand_output", True),
                    pre_norm_act=kwargs.get("conv_pre_norm_act", False),
                    attn_early=kwargs.get("conv_attn_early", False),
                    attn_act_layer=kwargs.get("conv_attn_act_layer"),
                    norm_act_layer=kwargs.get("conv_norm_act_layer")))
            elif block_type == 'T':
                rel_pos_cls = kwargs.get("rel_pos_cls")
                blocks.append(TransformerBlock2d(
                    dim=dim_in, dim_out=out_chs, stride=block_stride,
                    rel_pos_cls=rel_pos_cls,
                    dim_head=kwargs.get("dim_head", 32),
                    expand_first=kwargs.get("transformer_expand_first", True),
                    head_first=kwargs.get("head_first", True),
                    attn_bias=kwargs.get("transformer_attn_bias", True),
                    norm_layer=kwargs.get("transformer_norm_layer", "layernorm2d"),
                    norm_eps=kwargs.get("norm_eps", 1e-6),
                    attn_drop=kwargs.get("attn_drop", 0.0),
                    proj_drop=kwargs.get("proj_drop", 0.0),
                    shortcut_bias=kwargs.get("transformer_shortcut_bias", True),
                    pool_type=kwargs.get("conv_pool_type", "avg2"),
                    mlp_ratio=kwargs.get("mlp_ratio", 4.0),
                    act_layer=kwargs.get("act_layer"),
                    init_values=kwargs.get("init_values"),
                    drop_path=drop_path))
            elif block_type == 'M':
                blocks.append(MaxxVitBlock(
                    dim=dim_in, dim_out=out_chs, stride=block_stride,
                    window_size=window_size, grid_size=grid_size,
                    dim_head=kwargs.get("dim_head", 32),
                    head_first=kwargs.get("head_first", True),
                    act_layer=kwargs.get("act_layer"),
                    norm_eps=kwargs.get("norm_eps", 1e-5),
                    attn_drop=kwargs.get("attn_drop", 0.0),
                    proj_drop=kwargs.get("proj_drop", 0.0),
                    conv_norm_eps=kwargs.get("conv_norm_eps", 1e-3),
                    conv_expand_ratio=kwargs.get("conv_expand_ratio", 4.0),
                    conv_kernel_size=kwargs.get("conv_kernel_size", 3),
                    conv_group_size=kwargs.get("conv_group_size", 1),
                    conv_output_bias=kwargs.get("conv_output_bias", True),
                    conv_padding=kwargs.get("conv_padding", "same"),
                    conv_attn_ratio=kwargs.get("conv_attn_ratio", 0.25),
                    conv_pool_type=kwargs.get("conv_pool_type", "avg2"),
                    conv_stride_mode=kwargs.get("conv_stride_mode", "dw"),
                    conv_expand_output=kwargs.get("conv_expand_output", True),
                    conv_pre_norm_act=kwargs.get("conv_pre_norm_act", False),
                    conv_attn_early=kwargs.get("conv_attn_early", False),
                    conv_attn_act_layer=kwargs.get("conv_attn_act_layer"),
                    conv_norm_act_layer=kwargs.get("conv_norm_act_layer"),
                    mlp_ratio=kwargs.get("mlp_ratio", 4.0),
                    init_values=kwargs.get("init_values"),
                    drop_path=drop_path))
            else:
                raise ValueError(f"Unknown block_type: {block_type}")
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


# ---------------------------------------------------------------------------
# Stem and Head
# ---------------------------------------------------------------------------

class Stem(nn.Layer):
    """Stem: Conv-BN-Act -> Conv.

    timm: Stem
    """
    def __init__(self, in_chs, out_chs, bias=True,
                 norm_eps=1e-3, padding="same", act_layer=None):
        super().__init__()
        if not isinstance(out_chs, (list, tuple)):
            out_chs = to_2tuple(out_chs)
        self.conv1 = create_conv2d(in_chs, out_chs[0], 3, stride=2,
                                    padding=padding, bias=bias)
        self.norm1 = BatchNormAct2d(out_chs[0], eps=norm_eps, apply_act=True, act_layer=act_layer)
        self.conv2 = create_conv2d(out_chs[0], out_chs[1], 3, stride=1,
                                    padding=padding, bias=bias)
        self.out_chs = out_chs[-1]

    def forward(self, x):
        # timm: conv1 -> norm1 -> conv2
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        return x


class ClassifierHead(nn.Layer):
    """Simple classifier head: global_pool -> flatten -> drop -> fc.

    timm: ClassifierHead. Used by CoAtNet when head_hidden_size is None.
    """
    def __init__(self, in_features, num_classes, pool_type="avg",
                 drop_rate=0.0):
        super().__init__()
        self.in_features = in_features
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits=False):
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.drop(x)
        if pre_logits:
            return x
        x = self.fc(x)
        return x


class NormMlpClassifierHead(nn.Layer):
    """Classifier head: pool -> norm -> flatten -> fc -> tanh -> drop -> fc.

    timm: NormMlpClassifierHead. Used by MaxViT when head_hidden_size is set.
    """
    def __init__(self, in_features, num_classes, hidden_size,
                 pool_type="avg", drop_rate=0.0, norm_eps=1e-5):
        super().__init__()
        self.norm = LayerNorm2d(in_features, eps=norm_eps)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.flatten = nn.Flatten()
        self.pre_logits = nn.Sequential(OrderedDict([
            ("fc", nn.Linear(in_features, hidden_size)),
            ("act", nn.Tanh()),
        ]))
        self.dropout = nn.Dropout(drop_rate)
        self.fc = nn.Linear(hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, pre_logits=False):
        x = self.pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        if pre_logits:
            return x
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MaxxVit(nn.Layer):
    """Unified MaxxViT model supporting both MaxViT and CoAtNet architectures.

    timm: MaxxVit. block_type controls per-stage composition:
      MaxViT:  ('M','M','M','M') -- every stage is MbConv + window/grid attention
      CoAtNet: ('C','C','T','T') -- first two stages MbConv, last two Transformer
    """
    def __init__(self, cfg_embed_dim, cfg_depths, cfg_stem_width,
                 cfg_head_hidden_size, cfg_stem_bias=True,
                 block_type=('M', 'M', 'M', 'M'),
                 img_size=224, in_chans=3, num_classes=1000,
                 drop_path_rate=0.0,
                 # conv config
                 conv_norm_eps=1e-3, conv_act='gelu_tanh', conv_padding='same',
                 conv_expand_ratio=4.0, conv_kernel_size=3, conv_group_size=1,
                 conv_output_bias=True, conv_attn_ratio=0.25, conv_pool_type='avg2',
                 conv_stride_mode='dw', conv_expand_output=True,
                 conv_pre_norm_act=False, conv_attn_early=False,
                 conv_attn_act='silu',
                 # transformer config (defaults match timm MaxxVitTransformerCfg)
                 transformer_norm_eps=1e-6, transformer_act='gelu',
                 transformer_head_first=True, transformer_dim_head=32,
                 transformer_rel_pos_type='bias', transformer_rel_pos_dim=512,
                 transformer_expand_first=True, transformer_shortcut_bias=True,
                 transformer_norm_layer='layernorm2d', transformer_attn_bias=True,
                 mlp_ratio=4.0, init_values=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.num_features = cfg_embed_dim[-1]

        # Resolve conv activation layer class (paddle native)
        # paddle nn.GELU(approximate='tanh') == timm gelu_tanh
        if conv_act == 'silu':
            conv_act_layer = nn.Silu
        else:  # 'gelu_tanh' or default
            conv_act_layer = partial(nn.GELU, approximate='tanh')

        # SE activation layer class
        if conv_attn_act == 'relu':
            se_act_layer = nn.ReLU
        else:
            se_act_layer = nn.Silu

        # Transformer activation layer class (paddle native)
        if transformer_act == 'gelu_tanh':
            tf_act_layer = partial(nn.GELU, approximate='tanh')
        else:  # 'gelu'
            tf_act_layer = nn.GELU

        # Relative position class for transformer blocks
        def make_rel_pos_cls(feat_size_ws):
            if transformer_rel_pos_type == 'mlp':
                return partial(RelPosMlp, window_size=feat_size_ws,
                               hidden_dim=transformer_rel_pos_dim)
            elif transformer_rel_pos_type == 'bias':
                return partial(RelPosBias, window_size=feat_size_ws)
            elif transformer_rel_pos_type == 'bias_tf':
                return partial(RelPosBiasTf, window_size=feat_size_ws)
            return None

        # Window/grid size for MaxViT partition attention
        partition_ratio = 32
        window_size = (img_size[0] // partition_ratio, img_size[1] // partition_ratio)
        grid_size = window_size

        # Stem
        self.stem = Stem(in_chans, cfg_stem_width, bias=cfg_stem_bias,
                          norm_eps=conv_norm_eps, padding=conv_padding,
                          act_layer=conv_act_layer)
        feat_size = (img_size[0] // 2, img_size[1] // 2)
        in_chs = self.stem.out_chs

        # Stages
        num_stages = len(cfg_embed_dim)
        dpr = _calc_drop_path_rates(drop_path_rate, cfg_depths)
        stages = []
        for i in range(num_stages):
            stage_stride = 2
            out_chs = cfg_embed_dim[i]
            feat_size = ((feat_size[0] - 1) // stage_stride + 1,
                         (feat_size[1] - 1) // stage_stride + 1)

            bt = block_type[i] if isinstance(block_type, (tuple, list)) else block_type

            # rel_pos_cls is needed for 'T' blocks
            rel_pos_cls = None
            if bt == 'T':
                rel_pos_cls = make_rel_pos_cls(feat_size)

            stages.append(MaxxVitStage(
                in_chs, out_chs, depth=cfg_depths[i], stride=stage_stride,
                block_type=bt,
                feat_size=feat_size,
                window_size=window_size, grid_size=grid_size,
                # conv params
                conv_norm_eps=conv_norm_eps, conv_padding=conv_padding,
                conv_expand_ratio=conv_expand_ratio, conv_kernel_size=conv_kernel_size,
                conv_group_size=conv_group_size, conv_output_bias=conv_output_bias,
                conv_attn_ratio=conv_attn_ratio, conv_pool_type=conv_pool_type,
                conv_stride_mode=conv_stride_mode, conv_expand_output=conv_expand_output,
                conv_pre_norm_act=conv_pre_norm_act, conv_attn_early=conv_attn_early,
                conv_attn_act_layer=se_act_layer, conv_norm_act_layer=conv_act_layer,
                # transformer params
                dim_head=transformer_dim_head, head_first=transformer_head_first,
                act_layer=tf_act_layer, norm_eps=transformer_norm_eps,
                attn_drop=0.0, proj_drop=0.0,
                transformer_expand_first=transformer_expand_first,
                transformer_shortcut_bias=transformer_shortcut_bias,
                transformer_norm_layer=transformer_norm_layer,
                transformer_attn_bias=transformer_attn_bias,
                mlp_ratio=mlp_ratio, init_values=init_values,
                # for 'T' blocks
                rel_pos_cls=rel_pos_cls,
                drop_path_rates=dpr[i]))
            in_chs = out_chs
        self.stages = nn.Sequential(*stages)

        # Head
        # timm logic:
        #   if head_hidden_size: NormMlpClassifierHead (norm=Identity inside head)
        #   else: self.norm = LayerNorm2d/BatchNorm2d + ClassifierHead (no pre_logits MLP)
        # The final self.norm follows transformer_norm_layer (e.g. 'batchnorm2d' for coatnet_bn_0_rw).
        if cfg_head_hidden_size:
            self.norm = nn.Identity()
            self.head = NormMlpClassifierHead(
                self.num_features, num_classes, cfg_head_hidden_size,
                norm_eps=transformer_norm_eps)
        else:
            if transformer_norm_layer == "batchnorm2d":
                self.norm = nn.BatchNorm2D(self.num_features, epsilon=transformer_norm_eps)
            else:
                self.norm = LayerNorm2d(self.num_features, eps=transformer_norm_eps)
            self.head = ClassifierHead(self.num_features, num_classes)

    def forward_features(self, x):
        # timm: stem -> stages -> norm
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ---------------------------------------------------------------------------
# MaxViT TF model constructors
# ---------------------------------------------------------------------------

MAXVIT_TF_CONFIGS = {
    "maxvit_tiny_tf": dict(
        embed_dim=(64, 128, 256, 512), depths=(2, 2, 5, 2),
        stem_width=64, head_hidden_size=512, stem_bias=True,
        dim_head=32, drop_path_rate=0.2),
    "maxvit_small_tf": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 2, 5, 2),
        stem_width=64, head_hidden_size=768, stem_bias=True,
        dim_head=32, drop_path_rate=0.2),
    "maxvit_base_tf": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 6, 14, 2),
        stem_width=64, head_hidden_size=768, stem_bias=True,
        dim_head=32, drop_path_rate=0.2),
    "maxvit_large_tf": dict(
        embed_dim=(128, 256, 512, 1024), depths=(2, 6, 14, 2),
        stem_width=128, head_hidden_size=1024, stem_bias=True,
        dim_head=32, drop_path_rate=0.2),
    "maxvit_xlarge_tf": dict(
        embed_dim=(192, 384, 768, 1536), depths=(2, 6, 14, 2),
        stem_width=192, head_hidden_size=1536, stem_bias=True,
        dim_head=32, drop_path_rate=0.2),
}


def _get_size_from_model_name(name):
    """Extract image size from model name like 'maxvit_tiny_tf_224'."""
    parts = name.split("_")
    for p in reversed(parts):
        if p.isdigit() and int(p) >= 32:
            return int(p)
    return 224


def create_maxvit(model_name):
    """Create a MaxxVit model matching timm's maxvit_tf architecture.

    Uses _tf_cfg defaults: conv gelu_tanh/eps=1e-3/padding=same,
    transformer gelu_tanh/eps=1e-5/head_first=False/bias_tf.
    """
    base = model_name
    for suffix in (".in1k", ".in21k"):
        base = base.replace(suffix, "")
    size = _get_size_from_model_name(base)
    config_key = base.rsplit("_", 1)[0]
    if config_key not in MAXVIT_TF_CONFIGS:
        raise ValueError(f"Unknown maxvit config: {config_key}")

    cfg = MAXVIT_TF_CONFIGS[config_key]

    return MaxxVit(
        cfg_embed_dim=cfg["embed_dim"],
        cfg_depths=cfg["depths"],
        cfg_stem_width=cfg["stem_width"],
        cfg_head_hidden_size=cfg["head_hidden_size"],
        cfg_stem_bias=cfg["stem_bias"],
        block_type=('M', 'M', 'M', 'M'),
        img_size=size,
        num_classes=1000,
        drop_path_rate=cfg["drop_path_rate"],
        # TF-specific config (_tf_cfg)
        conv_norm_eps=1e-3, conv_act='gelu_tanh', conv_padding='same',
        conv_stride_mode='dw', conv_expand_output=True,
        conv_pre_norm_act=False, conv_attn_early=False,
        # transformer _tf_cfg: gelu_tanh, eps=1e-5, head_first=False, bias_tf
        transformer_norm_eps=1e-5, transformer_act='gelu_tanh',
        transformer_head_first=False, transformer_rel_pos_type='bias_tf',
        transformer_dim_head=cfg["dim_head"],
    )


# ---------------------------------------------------------------------------
# CoAtNet model constructors
# ---------------------------------------------------------------------------

def _rw_coat_cfg(
        stride_mode='pool', pool_type='avg2',
        conv_output_bias=False, conv_attn_early=False,
        conv_attn_act_layer='relu', conv_norm_layer='',
        transformer_shortcut_bias=True,
        transformer_norm_layer='layernorm2d',
        init_values=None, rel_pos_type='bias', rel_pos_dim=512):
    """RW CoAtNet configuration builder.

    Matches timm.models.maxxvit._rw_coat_cfg defaults:
      conv: pre_norm_act=True, expand_output=False, act='silu', stride_mode='pool'
      transformer: expand_first=False, shortcut_bias=True, rel_pos_type='bias'
    """
    return dict(
        conv_stride_mode=stride_mode,
        conv_pool_type=pool_type,
        conv_pre_norm_act=True,
        conv_expand_output=False,
        conv_output_bias=conv_output_bias,
        conv_attn_early=conv_attn_early,
        conv_attn_act=conv_attn_act_layer,
        conv_act='silu',
        conv_norm_eps=1e-5,
        transformer_expand_first=False,
        transformer_shortcut_bias=transformer_shortcut_bias,
        transformer_norm_layer=transformer_norm_layer,
        transformer_init_values=init_values,
        transformer_rel_pos_type=rel_pos_type,
        transformer_rel_pos_dim=rel_pos_dim,
        transformer_norm_eps=1e-6,
    )


def _rw_max_cfg(
        stride_mode='dw', pool_type='avg2',
        conv_output_bias=False, conv_attn_ratio=1 / 16,
        transformer_shortcut_bias=True,
        transformer_norm_layer='layernorm2d',
        init_values=None, rel_pos_type='bias', rel_pos_dim=512):
    """RW MaxViT-style configuration builder (used by some CoAtNet models).

    Matches timm.models.maxxvit._rw_max_cfg defaults:
      conv: expand_output=False, act='silu', pre_norm_act=False, attn_act='silu'
      transformer: expand_first=False
    """
    return dict(
        conv_stride_mode=stride_mode,
        conv_pool_type=pool_type,
        conv_pre_norm_act=False,
        conv_expand_output=False,
        conv_output_bias=conv_output_bias,
        conv_attn_ratio=conv_attn_ratio,
        conv_attn_act='silu',
        conv_act='silu',
        conv_norm_eps=1e-5,
        transformer_expand_first=False,
        transformer_shortcut_bias=transformer_shortcut_bias,
        transformer_norm_layer=transformer_norm_layer,
        transformer_init_values=init_values,
        transformer_rel_pos_type=rel_pos_type,
        transformer_rel_pos_dim=rel_pos_dim,
        transformer_norm_eps=1e-6,
    )


COATNET_CONFIGS = {
    "coatnet_0_rw": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 3, 7, 2),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            conv_attn_early=True, transformer_shortcut_bias=False)),
    "coatnet_1_rw": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 6, 14, 2),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            stride_mode='dw', conv_attn_early=True,
            transformer_shortcut_bias=False)),
    "coatnet_2_rw": dict(
        embed_dim=(128, 256, 512, 1024), depths=(2, 6, 14, 2),
        stem_width=(64, 128), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            stride_mode='dw', conv_attn_act_layer='silu')),
    "coatnet_bn_0_rw": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 3, 7, 2),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            stride_mode='dw', conv_attn_early=True,
            transformer_shortcut_bias=False,
            transformer_norm_layer='batchnorm2d')),
    "coatnet_nano_rw": dict(
        embed_dim=(64, 128, 256, 512), depths=(3, 4, 6, 3),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_max_cfg(
            stride_mode='pool', conv_output_bias=True, conv_attn_ratio=0.25)),
    "coatnet_rmlp_1_rw": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 6, 14, 2),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            pool_type='max', conv_attn_early=True,
            transformer_shortcut_bias=False,
            rel_pos_type='mlp', rel_pos_dim=384)),
    "coatnet_rmlp_1_rw2": dict(
        embed_dim=(96, 192, 384, 768), depths=(2, 6, 14, 2),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            stride_mode='dw', rel_pos_type='mlp', rel_pos_dim=512)),
    "coatnet_rmlp_2_rw": dict(
        embed_dim=(128, 256, 512, 1024), depths=(2, 6, 14, 2),
        stem_width=(64, 128), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_coat_cfg(
            stride_mode='dw', conv_attn_act_layer='silu',
            init_values=1e-6, rel_pos_type='mlp')),
    "coatnet_rmlp_nano_rw": dict(
        embed_dim=(64, 128, 256, 512), depths=(3, 4, 6, 3),
        stem_width=(32, 64), head_hidden_size=None, stem_bias=False,
        dim_head=32, drop_path_rate=0.0,
        **_rw_max_cfg(
            conv_output_bias=True, conv_attn_ratio=0.25,
            rel_pos_type='mlp', rel_pos_dim=384)),
}


def _parse_coatnet_name(model_name):
    """Parse coatnet model name to (config_key, img_size).

    Example: 'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k' -> ('coatnet_rmlp_2_rw', 384)
    """
    base = model_name
    for suffix in (".sw_in1k", ".sw_in12k_ft_in1k"):
        base = base.replace(suffix, "")
    parts = base.split("_")
    size = int(parts[-1])
    config_key = "_".join(parts[:-1])
    return config_key, size


def create_coatnet(model_name):
    """Create a MaxxVit model matching timm's coatnet architecture.

    Uses CoAtNet block_type=('C','C','T','T').
    CoAtNet uses symmetric padding ('' in timm MaxxVitConvCfg), unlike MaxViT TF.
    """
    config_key, size = _parse_coatnet_name(model_name)
    if config_key not in COATNET_CONFIGS:
        raise ValueError(f"Unknown coatnet config: {config_key}")

    cfg = COATNET_CONFIGS[config_key]

    # Extract CoAtNet-specific params from config
    kwargs = {}
    for k in ('conv_stride_mode', 'conv_pool_type', 'conv_pre_norm_act',
              'conv_expand_output', 'conv_output_bias', 'conv_attn_early',
              'conv_attn_act', 'conv_act', 'conv_norm_eps',
              'transformer_expand_first', 'transformer_shortcut_bias',
              'transformer_norm_layer', 'transformer_rel_pos_type',
              'transformer_rel_pos_dim', 'transformer_norm_eps'):
        if k in cfg:
            kwargs[k] = cfg[k]

    init_values = cfg.get('transformer_init_values')

    return MaxxVit(
        cfg_embed_dim=cfg["embed_dim"],
        cfg_depths=cfg["depths"],
        cfg_stem_width=cfg["stem_width"],
        cfg_head_hidden_size=cfg["head_hidden_size"],
        cfg_stem_bias=cfg["stem_bias"],
        block_type=('C', 'C', 'T', 'T'),
        img_size=size,
        num_classes=1000,
        drop_path_rate=cfg["drop_path_rate"],
        transformer_dim_head=cfg["dim_head"],
        init_values=init_values,
        conv_attn_ratio=cfg.get('conv_attn_ratio', 0.25),
        # CoAtNet uses symmetric PyTorch-style padding ('' in timm MaxxVitConvCfg).
        # MaxViT TF uses 'same' (TF-style asymmetric padding).
        conv_padding='',
        # CoAtNet transformer uses head_first=True (timm MaxxVitTransformerCfg default).
        # MaxViT TF uses head_first=False (_tf_cfg override).
        transformer_head_first=True,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# PaddleClas-style factory functions
# ---------------------------------------------------------------------------

def _make_maxvit_factory(config_key, img_size):
    # config_key='maxvit_tiny_tf', img_size=224 -> 'MaxViT_tiny_tf_224'
    # PaddleClas convention: keep suffix lowercase (cf. ConvNeXt_tiny, MobileNetV2_x0_25)
    suffix = config_key[len("maxvit_"):]
    name = f"MaxViT_{suffix}_{img_size}"

    def _factory(pretrained=False, **kwargs):
        # timm: create_maxvit(f"{config_key}_{img_size}")
        model_name = f"{config_key}_{img_size}"
        return create_maxvit(model_name)

    _factory.__name__ = name
    _factory.__qualname__ = name
    return _factory


def _make_coatnet_factory(config_key, img_size):
    # config_key='coatnet_0_rw', img_size=224 -> 'CoAtNet_0_rw_224'
    suffix = config_key[len("coatnet_"):]
    name = f"CoAtNet_{suffix}_{img_size}"

    def _factory(pretrained=False, **kwargs):
        # timm: create_coatnet(f"{config_key}_{img_size}")
        model_name = f"{config_key}_{img_size}"
        return create_coatnet(model_name)

    _factory.__name__ = name
    _factory.__qualname__ = name
    return _factory


# MaxViT factory functions (matching PaddleClas naming convention)
MaxViT_tiny_tf_224 = _make_maxvit_factory("maxvit_tiny_tf", 224)
MaxViT_tiny_tf_384 = _make_maxvit_factory("maxvit_tiny_tf", 384)
MaxViT_tiny_tf_512 = _make_maxvit_factory("maxvit_tiny_tf", 512)
MaxViT_small_tf_224 = _make_maxvit_factory("maxvit_small_tf", 224)
MaxViT_small_tf_384 = _make_maxvit_factory("maxvit_small_tf", 384)
MaxViT_small_tf_512 = _make_maxvit_factory("maxvit_small_tf", 512)
MaxViT_base_tf_224 = _make_maxvit_factory("maxvit_base_tf", 224)
MaxViT_base_tf_384 = _make_maxvit_factory("maxvit_base_tf", 384)
MaxViT_base_tf_512 = _make_maxvit_factory("maxvit_base_tf", 512)
MaxViT_large_tf_224 = _make_maxvit_factory("maxvit_large_tf", 224)
MaxViT_large_tf_384 = _make_maxvit_factory("maxvit_large_tf", 384)
MaxViT_large_tf_512 = _make_maxvit_factory("maxvit_large_tf", 512)

# CoAtNet factory functions
CoAtNet_0_rw_224 = _make_coatnet_factory("coatnet_0_rw", 224)
CoAtNet_1_rw_224 = _make_coatnet_factory("coatnet_1_rw", 224)
CoAtNet_2_rw_224 = _make_coatnet_factory("coatnet_2_rw", 224)
CoAtNet_bn_0_rw_224 = _make_coatnet_factory("coatnet_bn_0_rw", 224)
CoAtNet_nano_rw_224 = _make_coatnet_factory("coatnet_nano_rw", 224)
CoAtNet_rmlp_1_rw_224 = _make_coatnet_factory("coatnet_rmlp_1_rw", 224)
CoAtNet_rmlp_1_rw2_224 = _make_coatnet_factory("coatnet_rmlp_1_rw2", 224)
CoAtNet_rmlp_2_rw_224 = _make_coatnet_factory("coatnet_rmlp_2_rw", 224)
CoAtNet_rmlp_2_rw_384 = _make_coatnet_factory("coatnet_rmlp_2_rw", 384)
CoAtNet_rmlp_nano_rw_224 = _make_coatnet_factory("coatnet_rmlp_nano_rw", 224)
