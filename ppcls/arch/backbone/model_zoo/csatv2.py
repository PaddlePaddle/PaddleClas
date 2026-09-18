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
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/csatv2.py

"""CSATv2: frequency-domain vision model with DCT stem and spatial attention.
"""

import math
import warnings
from functools import reduce

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain
from ..base.theseus_layer import TheseusLayer
from .vision_transformer import DropPath, Mlp, trunc_normal_, zeros_, ones_

__all__ = ['CSATv2_512', 'CSATv2_21m_512', 'CSATv2_21m_640']

MODEL_URLS = {
    "CSATv2_512": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CSATv2_512_pretrained.pdparams"
    ),
    "CSATv2_21m_512": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CSATv2_21m_512_pretrained.pdparams"
    ),
    "CSATv2_21m_640": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/CSATv2_21m_640_pretrained.pdparams"
    ),
}

# DCT frequency normalization statistics (Y, Cb, Cr channels x 64 coefficients)
_DCT_MEAN = (
    (932.42657, -0.00260, 0.33415, -0.02840, 0.00003, -0.02792, -0.00183, 0.00006,
     0.00032, 0.03402, -0.00571, 0.00020, 0.00006, -0.00038, -0.00558, -0.00116,
     -0.00000, -0.00047, -0.00008, -0.00030, 0.00942, 0.00161, -0.00009, -0.00006,
     -0.00014, -0.00035, 0.00001, -0.00220, 0.00033, -0.00002, -0.00003, -0.00020,
     0.00007, -0.00000, 0.00005, 0.00293, -0.00004, 0.00006, 0.00019, 0.00004,
     0.00006, -0.00015, -0.00002, 0.00007, 0.00010, -0.00004, 0.00008, 0.00000,
     0.00008, -0.00001, 0.00015, 0.00002, 0.00007, 0.00003, 0.00004, -0.00001,
     0.00004, -0.00000, 0.00002, -0.00000, -0.00008, -0.00000, -0.00003, 0.00003),
    (962.34735, -0.00428, 0.09835, 0.00152, -0.00009, 0.00312, -0.00141, -0.00001,
     -0.00013, 0.01050, 0.00065, 0.00006, -0.00000, 0.00003, 0.00264, 0.00000,
     0.00001, 0.00007, -0.00006, 0.00003, 0.00341, 0.00163, 0.00004, 0.00003,
     -0.00001, 0.00008, -0.00000, 0.00090, 0.00018, -0.00006, -0.00001, 0.00007,
     -0.00003, -0.00001, 0.00006, 0.00084, -0.00000, -0.00001, 0.00000, 0.00004,
     -0.00001, -0.00002, 0.00000, 0.00001, 0.00002, 0.00001, 0.00004, 0.00011,
     0.00000, -0.00003, 0.00011, -0.00002, 0.00001, 0.00001, 0.00001, 0.00001,
     -0.00007, -0.00003, 0.00001, 0.00000, 0.00001, 0.00002, 0.00001, 0.00000),
    (1053.16101, -0.00213, -0.09207, 0.00186, 0.00013, 0.00034, -0.00119, 0.00002,
     0.00011, -0.00984, 0.00046, -0.00007, -0.00001, -0.00005, 0.00180, 0.00042,
     0.00002, -0.00010, 0.00004, 0.00003, -0.00301, 0.00125, -0.00002, -0.00003,
     -0.00001, -0.00001, -0.00001, 0.00056, 0.00021, 0.00001, -0.00001, 0.00002,
     -0.00001, -0.00001, 0.00005, -0.00070, -0.00002, -0.00002, 0.00005, -0.00004,
     -0.00000, 0.00002, -0.00002, 0.00001, 0.00000, -0.00003, 0.00004, 0.00007,
     0.00001, 0.00000, 0.00013, -0.00000, 0.00000, 0.00002, -0.00000, -0.00001,
     -0.00004, -0.00003, 0.00000, 0.00001, -0.00001, 0.00001, -0.00000, 0.00000),
)

_DCT_VAR = (
    (270372.37500, 6287.10645, 5974.94043, 1653.10889, 1463.91748, 1832.58997, 755.92468, 692.41528,
     648.57184, 641.46881, 285.79288, 301.62100, 380.43405, 349.84027, 374.15891, 190.30960,
     190.76746, 221.64578, 200.82646, 145.87979, 126.92046, 62.14622, 67.75562, 102.42001,
     129.74922, 130.04631, 103.12189, 97.76417, 53.17402, 54.81048, 73.48712, 81.04342,
     69.35100, 49.06024, 33.96053, 37.03279, 20.48858, 24.94830, 33.90822, 44.54912,
     47.56363, 40.03160, 30.43313, 22.63899, 26.53739, 26.57114, 21.84404, 17.41557,
     15.18253, 10.69678, 11.24111, 12.97229, 15.08971, 15.31646, 8.90409, 7.44213,
     6.66096, 6.97719, 4.17834, 3.83882, 4.51073, 2.36646, 2.41363, 1.48266),
    (18839.21094, 321.70932, 300.15259, 77.47830, 76.02293, 89.04748, 33.99642, 34.74807,
     32.12333, 28.19588, 12.04675, 14.26871, 18.45779, 16.59588, 15.67892, 7.37718,
     8.56312, 10.28946, 9.41013, 6.69090, 5.16453, 2.55186, 3.03073, 4.66765,
     5.85418, 5.74644, 4.33702, 3.66948, 1.95107, 2.26034, 3.06380, 3.50705,
     3.06359, 2.19284, 1.54454, 1.57860, 0.97078, 1.13941, 1.48653, 1.89996,
     1.95544, 1.64950, 1.24754, 0.93677, 1.09267, 1.09516, 0.94163, 0.78966,
     0.72489, 0.50841, 0.50909, 0.55664, 0.63111, 0.64125, 0.38847, 0.33378,
     0.30918, 0.33463, 0.20875, 0.19298, 0.21903, 0.13380, 0.13444, 0.09554),
    (17127.39844, 292.81421, 271.45209, 66.64056, 63.60253, 76.35437, 28.06587, 27.84831,
     25.96656, 23.60370, 9.99173, 11.34992, 14.46955, 12.92553, 12.69353, 5.91537,
     6.60187, 7.90891, 7.32825, 5.32785, 4.29660, 2.13459, 2.44135, 3.66021,
     4.50335, 4.38959, 3.34888, 2.97181, 1.60633, 1.77010, 2.35118, 2.69018,
     2.38189, 1.74596, 1.26014, 1.31684, 0.79327, 0.92046, 1.17670, 1.47609,
     1.50914, 1.28725, 0.99898, 0.74832, 0.85736, 0.85800, 0.74663, 0.63508,
     0.58748, 0.41098, 0.41121, 0.44663, 0.50277, 0.51519, 0.31729, 0.27336,
     0.25399, 0.27241, 0.17353, 0.16255, 0.18440, 0.11602, 0.11511, 0.08450),
)


class LayerNorm2d(nn.Layer):
    """Channel-normalizing LayerNorm for NCHW tensors, params named weight/bias to match timm."""

    def __init__(self, num_channels, epsilon=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[num_channels], default_initializer=ones_)
        self.bias = self.create_parameter(
            shape=[num_channels], default_initializer=zeros_)
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(2).mean(axis=1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.epsilon)
        return self.weight.reshape([1, -1, 1, 1]) * x + self.bias.reshape([1, -1, 1, 1])


class GlobalResponseNorm(nn.Layer):
    """Global Response Norm (ConvNeXt-V2), channels-last layout only (as used by CSATv2)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = self.create_parameter(
            shape=[dim], default_initializer=zeros_)
        self.bias = self.create_parameter(
            shape=[dim], default_initializer=zeros_)

    def forward(self, x):
        gx = paddle.sqrt((x * x).sum(axis=[1, 2], keepdim=True))
        nx = gx / (gx.mean(axis=-1, keepdim=True) + self.eps)
        return x + self.bias.reshape([1, 1, 1, -1]) + self.weight.reshape(
            [1, 1, 1, -1]) * (x * nx)


class Attention(nn.Layer):
    """timm-style multi-head attention with attn_head_dim decoupled from dim (qkv has no bias)."""

    def __init__(self, dim, num_heads=8, attn_head_dim=None, dim_out=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        dim_out = dim_out or dim
        head_dim = attn_head_dim if attn_head_dim is not None else dim // num_heads
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.attn_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, self.attn_dim * 3, bias_attr=False)
        self.attn_drop = nn.Dropout(p=attn_drop)
        self.proj = nn.Linear(self.attn_dim, dim_out)
        self.proj_drop = nn.Dropout(p=proj_drop)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(
            [B, N, 3, self.num_heads, self.head_dim]).transpose([2, 0, 3, 1, 4])
        q, k, v = paddle.unbind(qkv, axis=0)
        q = q * self.scale
        attn = q @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, self.attn_dim])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PosConv(nn.Layer):
    """Convolutional position encoding (depthwise 3x3 on the feature map)."""

    def __init__(self, in_chans):
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans,
            in_chans,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=True,
            groups=in_chans)

    def forward(self, x, size):
        B, N, C = x.shape
        H, W = size
        cnn_feat = x.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = self.proj(cnn_feat) + cnn_feat
        return x.flatten(2).transpose([0, 2, 1])


def _zigzag_permutation(rows, cols):
    """Generate zigzag scan order for DCT coefficients."""
    idx_matrix = np.arange(0, rows * cols, 1).reshape(rows, cols).tolist()
    dia = [[] for _ in range(rows + cols - 1)]
    zigzag = []
    for i in range(rows):
        for j in range(cols):
            s = i + j
            if s % 2 == 0:
                dia[s].insert(0, idx_matrix[i][j])
            else:
                dia[s].append(idx_matrix[i][j])
    for d in dia:
        zigzag.extend(d)
    return zigzag


def _dct_kernel_type_2(kernel_size):
    """Standard orthonormal Type-II DCT matrix.

    Equals the transpose of timm's fft-built kernel (verified to 6e-08), which
    is exactly what timm's Dct1d stores as its weights buffer.
    """
    n = np.arange(kernel_size, dtype='float64')
    k = n[:, None]
    c = np.cos(np.pi * k * (2 * n[None, :] + 1) / (2 * kernel_size))
    c[0] *= math.sqrt(1.0 / kernel_size)
    c[1:] *= math.sqrt(2.0 / kernel_size)
    return paddle.to_tensor(c.astype('float32'))


class Dct1d(nn.Layer):
    """1D Type-II DCT along the last dim; applies x @ weights.T (torch F.linear semantics)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.register_buffer('weights', _dct_kernel_type_2(kernel_size))

    def forward(self, x):
        return x @ self.weights.T


class Dct2d(nn.Layer):
    """2D DCT: 1D DCT along the last dim, then along the second-to-last dim."""

    def __init__(self, kernel_size):
        super().__init__()
        self.transform = Dct1d(kernel_size)

    def forward(self, x):
        perm = list(range(x.ndim))
        perm[-2], perm[-1] = perm[-1], perm[-2]
        return self.transform(self.transform(x).transpose(perm)).transpose(perm)


def _split_out_chs(out_chs, ratio=(24, 4, 4)):
    # reduce ratio to smallest integers (24,4,4) -> (6,1,1)
    g = reduce(math.gcd, ratio)
    r = tuple(x // g for x in ratio)
    denom = sum(r)
    assert out_chs % denom == 0 and out_chs >= denom, (
        f"out_chs={out_chs} can't be split into Y/Cb/Cr with ratio {ratio} "
        f"(reduced {r}); out_chs must be a multiple of {denom}.")
    unit = out_chs // denom
    y, cb, cr = (ri * unit for ri in r)
    assert y + cb + cr == out_chs and min(y, cb, cr) > 0
    return y, cb, cr


class LearnableDct2d(nn.Layer):
    """Learnable 2D DCT stem with RGB to YCbCr conversion and frequency selection."""

    def __init__(self, kernel_size, out_chs=32):
        super().__init__()
        self.k = kernel_size
        self.transform = Dct2d(kernel_size)
        self.permutation = _zigzag_permutation(kernel_size, kernel_size)
        y_ch, cb_ch, cr_ch = _split_out_chs(out_chs, ratio=(24, 4, 4))
        self.conv_y = nn.Conv2D(kernel_size**2, y_ch, kernel_size=1, padding=0)
        self.conv_cb = nn.Conv2D(kernel_size**2, cb_ch, kernel_size=1, padding=0)
        self.conv_cr = nn.Conv2D(kernel_size**2, cr_ch, kernel_size=1, padding=0)

        # non-persistent normalization statistics, rebuilt at init (not in checkpoints)
        self.register_buffer(
            'mean',
            paddle.to_tensor(np.asarray(_DCT_MEAN, dtype='float32')),
            persistable=False)
        self.register_buffer(
            'var',
            paddle.to_tensor(np.asarray(_DCT_VAR, dtype='float32')),
            persistable=False)
        self.register_buffer(
            'imagenet_mean',
            paddle.to_tensor([0.485, 0.456, 0.406]).reshape([3, 1, 1]),
            persistable=False)
        self.register_buffer(
            'imagenet_std',
            paddle.to_tensor([0.229, 0.224, 0.225]).reshape([3, 1, 1]),
            persistable=False)

    def _denormalize(self, x):
        # from ImageNet-normalized input back to [0, 255] range
        return (x * self.imagenet_std + self.imagenet_mean) * 255

    def _rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0], x[:, 1], x[:, 2]
        y = r * 0.299 + g * 0.587 + b * 0.114
        cb = 0.564 * (b - y) + 128
        cr = 0.713 * (r - y) + 128
        return paddle.stack([y, cb, cr], axis=1)

    def _frequency_normalize(self, x):
        std = self.var**0.5 + 1e-8
        return (x - self.mean) / std

    def forward(self, x):
        b, c, h, w = x.shape
        x = self._denormalize(x)
        x = self._rgb_to_ycbcr(x)
        # extract non-overlapping k x k patches
        x = x.reshape([b, c, h // self.k, self.k, w // self.k, self.k])
        x = x.permute([0, 2, 4, 1, 3, 5])
        x = self.transform(x)
        x = x.reshape([-1, c, self.k * self.k])
        x = x[:, :, self.permutation]
        x = self._frequency_normalize(x)
        x = x.reshape([b, h // self.k, w // self.k, c, -1])
        x = x.permute([0, 3, 4, 1, 2])
        x_y = self.conv_y(x[:, 0])
        x_cb = self.conv_cb(x[:, 1])
        x_cr = self.conv_cr(x[:, 2])
        return paddle.concat([x_y, x_cb, x_cr], axis=1)


class SpatialTransformerBlock(nn.Layer):
    """Lightweight single-head, 1-dim attention over spatial positions (7x7 grid)."""

    def __init__(self):
        super().__init__()
        self.pos_embed = PosConv(in_chans=1)
        self.norm1 = nn.LayerNorm(1, epsilon=1e-6)
        self.qkv = nn.Linear(1, 3, bias_attr=False)
        self.norm2 = nn.LayerNorm(1, epsilon=1e-6)
        self.mlp = Mlp(1, 4, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # attention block
        shortcut = x
        x_t = x.flatten(2).transpose([0, 2, 1])  # (B, N, 1)
        x_t = self.norm1(x_t)
        x_t = self.pos_embed(x_t, (H, W))
        qkv = self.qkv(x_t)  # (B, N, 3)
        q, k, v = paddle.unbind(qkv, axis=-1)  # each (B, N)
        # upstream quirk kept verbatim: 2D q/k yield a (B, B) cross-batch
        # attention matrix (timm's (B, N, N) comment is wrong); with B=1 this
        # degenerates to passing v through
        attn = F.softmax(q @ k.transpose([1, 0]), axis=-1)  # (B, B)
        x_t = (attn @ v).unsqueeze(-1)  # (B, N, 1)
        x_t = x_t.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = shortcut + x_t

        # feedforward block
        shortcut = x
        x_t = x.flatten(2).transpose([0, 2, 1])
        x_t = self.mlp(self.norm2(x_t))
        x_t = x_t.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = shortcut + x_t

        return x


class SpatialAttention(nn.Layer):
    """Spatial attention from channel statistics with a small transformer."""

    def __init__(self):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2D(7)
        self.conv = nn.Conv2D(2, 1, kernel_size=7, padding=3)
        self.attn = SpatialTransformerBlock()

    def forward(self, x):
        x_avg = x.mean(axis=1, keepdim=True)
        x_max = paddle.amax(x, axis=1, keepdim=True)
        x = paddle.concat([x_avg, x_max], axis=1)
        x = self.avgpool(x)
        x = self.conv(x)
        x = self.attn(x)
        return x


class Block(nn.Layer):
    """ConvNeXt-style block with spatial attention gating."""

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SpatialAttention()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.transpose([0, 3, 1, 2])

        attn = self.attn(x)
        attn = F.interpolate(
            attn, size=x.shape[2:], mode='bilinear', align_corners=True)
        x = x * attn

        return shortcut + self.drop_path(x)


class TransformerBlock(nn.Layer):
    """Transformer block with optional downsampling and convolutional position encoding."""

    def __init__(self, inp, oup, num_heads=8, attn_head_dim=32, downsample=False,
                 attn_drop=0., proj_drop=0., drop_path=0.):
        super().__init__()
        hidden_dim = int(inp * 4)
        self.downsample = downsample

        if self.downsample:
            self.pool1 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
            self.pool2 = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
            self.proj = nn.Conv2D(
                inp, oup, kernel_size=1, stride=1, padding=0, bias_attr=False)
        else:
            self.pool1 = nn.Identity()
            self.pool2 = nn.Identity()
            self.proj = nn.Identity()

        self.pos_embed = PosConv(in_chans=inp)
        self.norm1 = nn.LayerNorm(inp, epsilon=1e-6)
        self.attn = Attention(
            dim=inp,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            dim_out=oup,
            attn_drop=attn_drop,
            proj_drop=proj_drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(oup, epsilon=1e-6)
        self.mlp = Mlp(oup, hidden_dim, oup, drop=proj_drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.downsample:
            shortcut = self.proj(self.pool1(x))
            x_t = self.pool2(x)
            B, C, H, W = x_t.shape
        else:
            B, C, H, W = x.shape
            shortcut = x
            x_t = x
        x_t = x_t.flatten(2).transpose([0, 2, 1])
        x_t = self.norm1(x_t)
        x_t = self.pos_embed(x_t, (H, W))
        x_t = self.attn(x_t)
        x_t = x_t.transpose([0, 2, 1]).reshape([B, -1, H, W])
        x = shortcut + self.drop_path1(x_t)

        # MLP block
        B, C, H, W = x.shape
        shortcut = x
        x_t = x.flatten(2).transpose([0, 2, 1])
        x_t = self.mlp(self.norm2(x_t))
        x_t = x_t.transpose([0, 2, 1]).reshape([B, C, H, W])
        x = shortcut + self.drop_path2(x_t)

        return x


class NormMlpClassifierHead(nn.Layer):
    """Pool -> LayerNorm2d -> Flatten -> Dropout -> Linear classifier head."""

    def __init__(self, in_features, num_classes, pool_type='avg', drop_rate=0.):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2D(1) if pool_type else nn.Identity()
        self.norm = LayerNorm2d(in_features, epsilon=1e-6)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(in_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class CSATv2(TheseusLayer):
    """CSATv2: frequency-domain vision model with spatial attention.

    A hybrid architecture that processes images in the DCT frequency domain
    with ConvNeXt-style blocks and transformer attention.
    """

    def __init__(self,
                 num_classes=1000,
                 in_chans=3,
                 dims=(32, 72, 168, 386),
                 depths=(2, 2, 8, 6),
                 transformer_depths=(0, 0, 2, 2),
                 drop_path_rate=0.,
                 transformer_drop_path=False,
                 global_pool='avg'):
        super().__init__()
        if in_chans != 3:
            warnings.warn(
                'CSATv2 is designed for 3-channel RGB input. '
                'in_chans={} may not work correctly with the DCT stem.'.format(in_chans))
        self.num_classes = num_classes
        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_features = dims[-1]

        # transformer blocks keep drop_path=0 unless transformer_drop_path is set,
        # but still occupy their linspace slots when it is set
        total_blocks = sum(depths) if transformer_drop_path else sum(
            d - t for d, t in zip(depths, transformer_depths))
        dp_iter = iter(paddle.linspace(0, drop_path_rate, total_blocks).numpy().tolist())
        dp_rates = []
        for depth, t_depth in zip(depths, transformer_depths):
            dp_rates += [next(dp_iter) for _ in range(depth - t_depth)]
            dp_rates += [next(dp_iter) if transformer_drop_path else 0. for _ in range(t_depth)]

        self.stem_dct = LearnableDct2d(8, out_chs=dims[0])

        dp_iter = iter(dp_rates)
        stages = []
        for i, (dim, depth, t_depth) in enumerate(zip(dims, depths, transformer_depths)):
            layers = ([nn.Conv2D(dims[i - 1], dim, kernel_size=2, stride=2)] if i > 0 else []) + \
                [Block(dim=dim, drop_path=next(dp_iter))
                 for _ in range(depth - t_depth)] + \
                [TransformerBlock(inp=dim, oup=dim,
                                  drop_path=next(dp_iter) if transformer_drop_path else 0.)
                 for _ in range(t_depth)] + \
                ([LayerNorm2d(dim, epsilon=1e-6)] if i < len(depths) - 1 else [])
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)

        self.head = NormMlpClassifierHead(dims[-1], num_classes, pool_type=global_pool)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_features(self, x):
        x = self.stem_dct(x)
        x = self.stages(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
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
            "pretrained type is not available. Please use `string` or `boolean` type.")


def CSATv2_512(pretrained=False, use_ssld=False, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model = CSATv2(dims=(32, 72, 168, 386), depths=(2, 2, 8, 6),
                   transformer_depths=(0, 0, 2, 2), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["CSATv2_512"], use_ssld=use_ssld)
    return model


def CSATv2_21m_512(pretrained=False, use_ssld=False, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model = CSATv2(dims=(48, 96, 224, 448), depths=(3, 3, 10, 8),
                   transformer_depths=(0, 0, 4, 3), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["CSATv2_21m_512"], use_ssld=use_ssld)
    return model


def CSATv2_21m_640(pretrained=False, use_ssld=False, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model = CSATv2(dims=(48, 96, 224, 448), depths=(3, 3, 10, 8),
                   transformer_depths=(0, 0, 4, 3), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["CSATv2_21m_640"], use_ssld=use_ssld)
    return model
