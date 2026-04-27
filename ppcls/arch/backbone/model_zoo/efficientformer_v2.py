# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

import math
from functools import partial
from typing import Dict, List, Optional, Tuple, Type, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain


MODEL_URLS = {
    "EfficientFormerV2_L":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientformerv2_l.pdparams",
    "EfficientFormerV2_S0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientformerv2_s0.pdparams",
    "EfficientFormerV2_S1":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientformerv2_s1.pdparams",
    "EfficientFormerV2_S2":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientformerv2_s2.pdparams",
}

__all__ = list(MODEL_URLS.keys())


EfficientFormer_width = {
    "L": (40, 80, 192, 384),
    "S2": (32, 64, 144, 288),
    "S1": (32, 48, 120, 224),
    "S0": (32, 48, 96, 176),
}

EfficientFormer_depth = {
    "L": (5, 5, 15, 10),
    "S2": (4, 4, 12, 8),
    "S1": (3, 3, 9, 6),
    "S0": (2, 2, 6, 4),
}

EfficientFormer_expansion_ratios = {
    "L": (
        4,
        4,
        (4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4),
        (4, 4, 4, 3, 3, 3, 3, 4, 4, 4),
    ),
    "S2": (4, 4, (4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4), (4, 4, 3, 3, 3, 3, 4, 4)),
    "S1": (4, 4, (4, 4, 3, 3, 3, 3, 4, 4, 4), (4, 4, 3, 3, 4, 4)),
    "S0": (4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
}


def to_2tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x)


def to_ntuple(n):
    def parse(x):
        if isinstance(x, (list, tuple)):
            if len(x) == n:
                return tuple(x)
            elif len(x) == 1 and isinstance(x[0], (list, tuple)):
                return to_ntuple(n)(x[0])
            else:
                raise ValueError(f"Expected {n} values, got {len(x)}")
        return tuple([x] * n)

    return parse


def ndgrid(*tensors):
    if len(tensors) == 2:
        # Paddle meshgrid only supports >= 3D for indexing='ij'
        x, y = tensors
        nx = len(x)
        ny = len(y)
        grid_x = x.unsqueeze(1).expand([nx, ny])
        grid_y = y.unsqueeze(0).expand([nx, ny])
        return [grid_x, grid_y]
    return paddle.meshgrid(*tensors, indexing="ij")


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    return (x / keep_prob) * random_tensor


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerScale2d(nn.Layer):
    def __init__(self, dim, init_values=1e-5):
        super(LayerScale2d, self).__init__()
        self.gamma = self.create_parameter(
            shape=[dim, 1, 1], default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x):
        return x * self.gamma


class ConvNormAct(nn.Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding="",
        dilation=1,
        groups=1,
        bias=True,
        norm_layer="batchnorm2d",
        act_layer="gelu",
    ):
        super(ConvNormAct, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding if isinstance(padding, int) else 0,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )
        if isinstance(norm_layer, partial):
            self.norm = norm_layer(out_channels)
        elif isinstance(norm_layer, type):
            self.norm = norm_layer(out_channels)
        elif norm_layer == "batchnorm2d":
            self.norm = nn.BatchNorm2D(out_channels)
        else:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")
        if isinstance(act_layer, type):
            self.act = act_layer()
        elif act_layer == "gelu":
            self.act = nn.GELU()
        elif act_layer == "relu":
            self.act = nn.ReLU()
        elif act_layer == "hardswish":
            self.act = nn.Hardswish()
        else:
            raise ValueError(f"Unsupported act_layer: {act_layer}")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


def create_norm_layer(norm_layer, num_channels, eps=1e-5):
    if norm_layer == "batchnorm2d":
        return nn.BatchNorm2D(num_channels, epsilon=eps)
    if norm_layer == "identity":
        return nn.Identity()
    return norm_layer(num_channels)


class ConvNorm(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: Union[int, str] = "",
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_layer: str = "batchnorm2d",
        norm_kwargs: Optional[Dict] = None,
    ):
        norm_kwargs = norm_kwargs or {}
        super().__init__()
        if isinstance(padding, (tuple, int)):
            conv_padding = padding
        else:
            conv_padding = 0
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )
        eps = norm_kwargs.get("eps", 1e-5)
        self.bn = create_norm_layer(norm_layer, out_channels, eps=eps)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Attention2d(nn.Layer):
    attention_bias_cache: Dict[str, paddle.Tensor]

    def __init__(
        self,
        dim: int = 384,
        key_dim: int = 32,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: Union[int, Tuple[int, int]] = 7,
        act_layer: Type[nn.Layer] = nn.GELU,
        stride: Optional[int] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.stride = stride

        resolution = to_2tuple(resolution)
        self.input_resolution = resolution
        if stride is not None:
            resolution = tuple([math.ceil(r / stride) for r in resolution])
            self.stride_conv = ConvNorm(
                dim, dim, kernel_size=3, stride=stride, padding=1, groups=dim
            )
            # align_corners=False to match timm behavior
            self.upsample = nn.Upsample(
                size=None, scale_factor=stride, mode="bilinear", align_corners=False
            )
        else:
            self.stride_conv = None
            self.upsample = None

        self.resolution = resolution
        self.N = self.resolution[0] * self.resolution[1]
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        kh = self.key_dim * self.num_heads

        self.q = ConvNorm(dim, kh)
        self.k = ConvNorm(dim, kh)
        self.v = ConvNorm(dim, self.dh)
        self.v_local = ConvNorm(
            self.dh, self.dh, kernel_size=3, padding=1, groups=self.dh
        )
        self.talking_head1 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1)
        self.talking_head2 = nn.Conv2D(self.num_heads, self.num_heads, kernel_size=1)

        self.act = act_layer()
        self.proj = ConvNorm(self.dh, dim, 1)

        self.attention_biases = self.create_parameter(
            shape=[self.num_heads, self.N],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.attention_bias_idxs = None
        self.attention_bias_cache = {}

        self._init_buffers()

    def train(self):
        super().train()
        if self.attention_bias_cache:
            self.attention_bias_cache = {}

    def reset_parameters(self) -> None:
        pass

    def _compute_attention_bias_idxs(self):
        pos = paddle.stack(
            ndgrid(
                paddle.arange(self.resolution[0], dtype="int64"),
                paddle.arange(self.resolution[1], dtype="int64"),
            )
        ).flatten(1)
        rel_pos = (pos[:, :, None] - pos[:, None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        return rel_pos

    def _init_buffers(self) -> None:
        self.attention_bias_idxs = self._compute_attention_bias_idxs()
        self.attention_bias_cache = {}

    def init_non_persistent_buffers(self) -> None:
        self._init_buffers()

    def get_attention_biases(self) -> paddle.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        device_key = str(paddle.device.get_device())
        if device_key not in self.attention_bias_cache:
            self.attention_bias_cache[device_key] = self.attention_biases[
                :, self.attention_bias_idxs
            ]
        return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.stride_conv is not None:
            x = self.stride_conv(x)

        q = self.q(x).reshape([B, self.num_heads, -1, self.N]).transpose([0, 1, 3, 2])
        k = self.k(x).reshape([B, self.num_heads, -1, self.N]).transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape([B, self.num_heads, -1, self.N]).transpose([0, 1, 3, 2])

        attn = (q @ k) * self.scale
        attn = attn + self.get_attention_biases()
        attn = self.talking_head1(attn)
        attn = F.softmax(attn, axis=-1)
        attn = self.talking_head2(attn)

        x = (attn @ v).transpose([0, 1, 3, 2])
        x = x.reshape([B, self.dh, self.resolution[0], self.resolution[1]]) + v_local
        if self.upsample is not None:
            x = self.upsample(x)

        x = self.act(x)
        x = self.proj(x)
        return x


class LocalGlobalQuery(nn.Layer):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
    ):
        super().__init__()
        self.pool = nn.AvgPool2D(kernel_size=1, stride=2, padding=0)
        self.local = nn.Conv2D(
            in_dim, in_dim, kernel_size=3, stride=2, padding=1, groups=in_dim
        )
        self.proj = ConvNorm(in_dim, out_dim, 1)

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class Attention2dDownsample(nn.Layer):
    attention_bias_cache: Dict[str, paddle.Tensor]

    def __init__(
        self,
        dim: int = 384,
        key_dim: int = 16,
        num_heads: int = 8,
        attn_ratio: int = 4,
        resolution: Union[int, Tuple[int, int]] = 7,
        out_dim: Optional[int] = None,
        act_layer: Type[nn.Layer] = nn.GELU,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.resolution = to_2tuple(resolution)
        self.resolution2 = tuple([math.ceil(r / 2) for r in self.resolution])
        self.N = self.resolution[0] * self.resolution[1]
        self.N2 = self.resolution2[0] * self.resolution2[1]

        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.out_dim = out_dim or dim
        kh = self.key_dim * self.num_heads

        self.q = LocalGlobalQuery(dim, kh)
        self.k = ConvNorm(dim, kh, 1)
        self.v = ConvNorm(dim, self.dh, 1)
        self.v_local = ConvNorm(
            self.dh, self.dh, kernel_size=3, stride=2, padding=1, groups=self.dh
        )

        self.act = act_layer()
        self.proj = ConvNorm(self.dh, self.out_dim, 1)

        self.attention_biases = self.create_parameter(
            shape=[self.num_heads, self.N],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.attention_bias_idxs = None
        self.attention_bias_cache = {}

        self._init_buffers()

    def train(self):
        super().train()
        if self.attention_bias_cache:
            self.attention_bias_cache = {}

    def reset_parameters(self) -> None:
        pass

    def _compute_attention_bias_idxs(self):
        k_pos = paddle.stack(
            ndgrid(
                paddle.arange(self.resolution[0], dtype="int64"),
                paddle.arange(self.resolution[1], dtype="int64"),
            )
        ).flatten(1)
        q_pos = paddle.stack(
            ndgrid(
                paddle.arange(0, self.resolution[0], step=2, dtype="int64"),
                paddle.arange(0, self.resolution[1], step=2, dtype="int64"),
            )
        ).flatten(1)
        rel_pos = (q_pos[:, :, None] - k_pos[:, None, :]).abs()
        rel_pos = (rel_pos[0] * self.resolution[1]) + rel_pos[1]
        return rel_pos

    def _init_buffers(self) -> None:
        self.attention_bias_idxs = self._compute_attention_bias_idxs()
        self.attention_bias_cache = {}

    def init_non_persistent_buffers(self) -> None:
        self._init_buffers()

    def get_attention_biases(self) -> paddle.Tensor:
        if self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        device_key = str(paddle.device.get_device())
        if device_key not in self.attention_bias_cache:
            self.attention_bias_cache[device_key] = self.attention_biases[
                :, self.attention_bias_idxs
            ]
        return self.attention_bias_cache[device_key]

    def forward(self, x):
        B, C, H, W = x.shape

        q = self.q(x).reshape([B, self.num_heads, -1, self.N2]).transpose([0, 1, 3, 2])
        k = self.k(x).reshape([B, self.num_heads, -1, self.N]).transpose([0, 1, 2, 3])
        v = self.v(x)
        v_local = self.v_local(v)
        v = v.reshape([B, self.num_heads, -1, self.N]).transpose([0, 1, 3, 2])

        attn = (q @ k) * self.scale
        attn = attn + self.get_attention_biases()
        attn = F.softmax(attn, axis=-1)

        x = (attn @ v).transpose([0, 1, 3, 2])
        x = x.reshape([B, self.dh, self.resolution2[0], self.resolution2[1]]) + v_local
        x = self.act(x)
        x = self.proj(x)
        return x


class Downsample(nn.Layer):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 2,
        padding: Union[int, Tuple[int, int]] = 1,
        resolution: Union[int, Tuple[int, int]] = 7,
        use_attn: bool = False,
        act_layer: Type[nn.Layer] = nn.GELU,
        norm_layer: Optional[Type[nn.Layer]] = None,
    ):
        super().__init__()

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        if norm_layer is None:
            norm_layer = "identity"
        self.conv = ConvNorm(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            norm_layer=norm_layer,
        )

        if use_attn:
            self.attn = Attention2dDownsample(
                dim=in_chs,
                out_dim=out_chs,
                resolution=resolution,
                act_layer=act_layer,
            )
        else:
            self.attn = None

    def forward(self, x):
        out = self.conv(x)
        if self.attn is not None:
            return self.attn(x) + out
        return out


class ConvMlpWithNorm(nn.Layer):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[nn.Layer] = nn.GELU,
        norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
        drop: float = 0.0,
        mid_conv: bool = False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = ConvNormAct(
            in_features,
            hidden_features,
            1,
            bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        if mid_conv:
            self.mid = ConvNormAct(
                hidden_features,
                hidden_features,
                3,
                padding=1,
                groups=hidden_features,
                bias=True,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        else:
            self.mid = nn.Identity()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = ConvNorm(hidden_features, out_features, 1, norm_layer=norm_layer)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.mid(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class EfficientFormerV2Block(nn.Layer):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: Type[nn.Layer] = nn.GELU,
        norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
        proj_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        resolution: Union[int, Tuple[int, int]] = 7,
        stride: Optional[int] = None,
        use_attn: bool = True,
    ):
        super().__init__()

        if use_attn:
            self.token_mixer = Attention2d(
                dim,
                resolution=resolution,
                act_layer=act_layer,
                stride=stride,
            )
            self.ls1 = (
                LayerScale2d(dim, layer_scale_init_value)
                if layer_scale_init_value is not None
                else nn.Identity()
            )
            self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        else:
            self.token_mixer = None
            self.ls1 = None
            self.drop_path1 = None

        self.mlp = ConvMlpWithNorm(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop=proj_drop,
            mid_conv=True,
        )
        self.ls2 = (
            LayerScale2d(dim, layer_scale_init_value)
            if layer_scale_init_value is not None
            else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.token_mixer is not None:
            x = x + self.drop_path1(self.ls1(self.token_mixer(x)))
        x = x + self.drop_path2(self.ls2(self.mlp(x)))
        return x


class Stem4(nn.Layer):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        act_layer: Type[nn.Layer] = nn.GELU,
        norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
    ):
        super().__init__()
        self.stride = 4
        self.conv1 = ConvNormAct(
            in_chs,
            out_chs // 2,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            norm_layer="batchnorm2d",
            act_layer="gelu",
        )
        self.conv2 = ConvNormAct(
            out_chs // 2,
            out_chs,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=True,
            norm_layer="batchnorm2d",
            act_layer="gelu",
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EfficientFormerV2Stage(nn.Layer):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        depth: int,
        resolution: Union[int, Tuple[int, int]] = 7,
        downsample: bool = True,
        block_stride: Optional[int] = None,
        downsample_use_attn: bool = False,
        block_use_attn: bool = False,
        num_vit: int = 1,
        mlp_ratio: Union[float, Tuple[float, ...]] = 4.0,
        proj_drop: float = 0.0,
        drop_path: Union[float, List[float]] = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        act_layer: Type[nn.Layer] = nn.GELU,
        norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
    ):
        super().__init__()
        mlp_ratio = to_ntuple(depth)(mlp_ratio)
        resolution = to_2tuple(resolution)

        if downsample:
            self.downsample = Downsample(
                dim,
                dim_out,
                use_attn=downsample_use_attn,
                resolution=resolution,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            dim = dim_out
            resolution = tuple([math.ceil(r / 2) for r in resolution])
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            remain_idx = depth - num_vit - 1
            b = EfficientFormerV2Block(
                dim,
                resolution=resolution,
                stride=block_stride,
                mlp_ratio=mlp_ratio[block_idx],
                use_attn=block_use_attn and block_idx > remain_idx,
                proj_drop=proj_drop,
                drop_path=drop_path[block_idx],
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
            )
            blocks.append(b)
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class EfficientFormerV2(nn.Layer):
    def __init__(
        self,
        depths: Tuple[int, ...],
        in_chans: int = 3,
        img_size: Union[int, Tuple[int, int]] = 224,
        global_pool: str = "avg",
        embed_dims: Optional[Tuple[int, ...]] = None,
        downsamples: Optional[Tuple[bool, ...]] = None,
        mlp_ratios: Union[float, Tuple[float, ...], Tuple[Tuple[float, ...], ...]] = 4,
        norm_layer: str = "batchnorm2d",
        norm_eps: float = 1e-5,
        act_layer: str = "gelu",
        num_classes: int = 1000,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = 1e-5,
        num_vit: int = 0,
        distillation: bool = True,
    ):
        super().__init__()
        assert global_pool in ("avg", "")
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.feature_info = []
        img_size = to_2tuple(img_size)
        if norm_layer == "batchnorm2d":
            norm_layer_fn = partial(nn.BatchNorm2D, epsilon=norm_eps)
        elif isinstance(norm_layer, type):
            if norm_layer == nn.BatchNorm2D:
                norm_layer_fn = partial(nn.BatchNorm2D, epsilon=norm_eps)
            else:
                norm_layer_fn = norm_layer
        else:
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")

        if act_layer == "gelu":
            act_layer_fn = nn.GELU
        else:
            raise ValueError(f"Unsupported act_layer: {act_layer}")

        self.stem = Stem4(
            in_chans, embed_dims[0], act_layer=act_layer_fn, norm_layer=norm_layer_fn
        )
        prev_dim = embed_dims[0]
        stride = 4

        num_stages = len(depths)

        if isinstance(drop_path_rate, (float, int)):
            drop_path_rate = [drop_path_rate] * num_stages

        dpr = []
        for stage_idx in range(num_stages):
            stage_depth = depths[stage_idx]
            stage_dpr = [
                drop_path_rate[stage_idx] / stage_depth * (i + 1)
                for i in range(stage_depth)
            ]
            dpr.append(stage_dpr)

        downsamples = downsamples or (False,) + (True,) * (len(depths) - 1)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        stages = []
        for i in range(num_stages):
            curr_resolution = tuple([math.ceil(s / stride) for s in img_size])
            stage = EfficientFormerV2Stage(
                prev_dim,
                embed_dims[i],
                depth=depths[i],
                resolution=curr_resolution,
                downsample=downsamples[i],
                block_stride=2 if i == 2 else None,
                downsample_use_attn=i >= 3,
                block_use_attn=i >= 2,
                num_vit=num_vit,
                mlp_ratio=mlp_ratios[i],
                proj_drop=proj_drop_rate,
                drop_path=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                act_layer=act_layer_fn,
                norm_layer=norm_layer_fn,
            )
            if downsamples[i]:
                stride *= 2
            prev_dim = embed_dims[i]
            self.feature_info.append(
                dict(num_chs=prev_dim, reduction=stride, module=f"stages.{i}")
            )
            stages.append(stage)
        self.stages = nn.Sequential(*stages)

        self.num_features = self.head_hidden_size = embed_dims[-1]
        if isinstance(norm_layer_fn, partial):
            self.norm = norm_layer_fn(embed_dims[-1])
        else:
            self.norm = norm_layer_fn(embed_dims[-1], epsilon=norm_eps)
        self.head_drop = nn.Dropout(drop_rate)
        self.head = (
            nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        )
        self.dist = distillation
        if self.dist:
            self.head_dist = (
                nn.Linear(embed_dims[-1], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
        else:
            self.head_dist = None

        self.distilled_training = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.initializer.TruncatedNormal(std=0.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0.0)(m.bias)

    def init_weights(self):
        self.apply(self._init_weights)

    def get_classifier(self):
        return self.head, self.head_dist

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None):
        self.num_classes = num_classes
        if global_pool is not None:
            self.global_pool = global_pool
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.head_dist = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

    def set_distilled_training(self, enable=True):
        self.distilled_training = enable

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        if self.global_pool == "avg":
            x = x.mean(axis=[2, 3])
        x = self.head_drop(x)
        if pre_logits:
            return x
        x, x_dist = self.head(x), self.head_dist(x)
        if self.distilled_training and self.training:
            return x, x_dist
        else:
            # average the classifier predictions during inference
            return (x + x_dist) / 2

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


def efficientformerv2_s0(pretrained=False, use_ssld=False, **kwargs) -> EfficientFormerV2:
    model_args = dict(
        depths=EfficientFormer_depth["S0"],
        embed_dims=EfficientFormer_width["S0"],
        num_vit=2,
        drop_path_rate=0.0,
        mlp_ratios=EfficientFormer_expansion_ratios["S0"],
    )
    model = EfficientFormerV2(**dict(model_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientFormerV2_S0"], use_ssld=use_ssld)
    return model


def efficientformerv2_s1(pretrained=False, use_ssld=False, **kwargs) -> EfficientFormerV2:
    model_args = dict(
        depths=EfficientFormer_depth["S1"],
        embed_dims=EfficientFormer_width["S1"],
        num_vit=2,
        drop_path_rate=0.0,
        mlp_ratios=EfficientFormer_expansion_ratios["S1"],
    )
    model = EfficientFormerV2(**dict(model_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientFormerV2_S1"], use_ssld=use_ssld)
    return model


def efficientformerv2_s2(pretrained=False, use_ssld=False, **kwargs) -> EfficientFormerV2:
    model_args = dict(
        depths=EfficientFormer_depth["S2"],
        embed_dims=EfficientFormer_width["S2"],
        num_vit=4,
        drop_path_rate=0.02,
        mlp_ratios=EfficientFormer_expansion_ratios["S2"],
    )
    model = EfficientFormerV2(**dict(model_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientFormerV2_S2"], use_ssld=use_ssld)
    return model


def efficientformerv2_l(pretrained=False, use_ssld=False, **kwargs) -> EfficientFormerV2:
    model_args = dict(
        depths=EfficientFormer_depth["L"],
        embed_dims=EfficientFormer_width["L"],
        num_vit=6,
        drop_path_rate=0.1,
        mlp_ratios=EfficientFormer_expansion_ratios["L"],
    )
    model = EfficientFormerV2(**dict(model_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientFormerV2_L"], use_ssld=use_ssld)
    return model
