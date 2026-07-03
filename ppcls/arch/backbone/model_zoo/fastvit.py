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

# Code was based on https://github.com/apple/ml-fastvit
# reference: https://arxiv.org/abs/2303.14189

from functools import partial
from typing import List, Optional, Tuple, Type, Union

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "FastViT_T8": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_T8.pdparams",
    "FastViT_T12": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_T12.pdparams",
    "FastViT_SA12": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_SA12.pdparams",
    "FastViT_SA24": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_SA24.pdparams",
    "FastViT_SA36": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_SA36.pdparams",
    "FastViT_MA36": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/FastViT_MA36.pdparams"
}

__all__ = ["FastViT_T8", "FastViT_T12", "FastViT_SA12", "FastViT_SA24", "FastViT_SA36", "FastViT_MA36"]


def num_groups(group_size, channels):
    if not group_size:
        return 1
    else:
        assert channels % group_size == 0
        return channels // group_size


class SqueezeExcite(nn.Layer):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_chs, rd_ratio=0.25, rd_channels=None, rd_divisor=4):
        super().__init__()
        # If rd_channels is provided, use it directly; otherwise calculate from rd_ratio
        if rd_channels is None:
            rd_channels = int(in_chs * rd_ratio)
            rd_channels = max(rd_channels, rd_divisor)
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.fc1 = nn.Conv2D(in_chs, rd_channels, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2D(rd_channels, in_chs, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.gate(out)
        return x * out


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (len(x.shape) - 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
        random_tensor = paddle.floor(random_tensor)
        return x.divide(paddle.to_tensor(keep_prob, dtype=x.dtype)) * random_tensor


class ConvNormAct(nn.Layer):
    """Convolution + Normalization + Activation."""

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=None,
            groups=1,
            bias=False,
            norm_layer=nn.BatchNorm2D,
            act_layer=nn.GELU,
            apply_act=True,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups=groups,
            bias_attr=bias)
        self.bn = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer() if apply_act and act_layer else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class MobileOneBlock(nn.Layer):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            inference_mode: bool = False,
            use_se: bool = False,
            use_act: bool = True,
            use_scale_branch: bool = True,
            num_conv_branches: int = 1,
            act_layer: Type[nn.Layer] = nn.GELU,
    ) -> None:
        super().__init__()
        self.inference_mode = inference_mode
        self.groups = num_groups(group_size, in_chs)
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.num_conv_branches = num_conv_branches

        self.se = SqueezeExcite(out_chs, rd_ratio=0.0625, rd_divisor=1) if use_se else nn.Identity()

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                dilation=dilation,
                groups=self.groups,
                bias_attr=True,
            )
        else:
            self.reparam_conv = None

            self.identity = (
                nn.BatchNorm2D(num_features=in_chs)
                if out_chs == in_chs and stride == 1
                else None
            )

            if num_conv_branches > 0:
                self.conv_kxk = nn.LayerList([
                    ConvNormAct(
                        self.in_chs,
                        self.out_chs,
                        kernel_size=kernel_size,
                        stride=self.stride,
                        padding=kernel_size // 2,
                        groups=self.groups,
                        apply_act=False,
                    ) for _ in range(self.num_conv_branches)
                ])
            else:
                self.conv_kxk = None

            self.conv_scale = None
            if kernel_size > 1 and use_scale_branch:
                self.conv_scale = ConvNormAct(
                    self.in_chs,
                    self.out_chs,
                    kernel_size=1,
                    stride=self.stride,
                    groups=self.groups,
                    apply_act=False,
                )

        self.act = act_layer() if use_act else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.reparam_conv is not None:
            return self.act(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.identity is not None:
            identity_out = self.identity(x)

        scale_out = 0
        if self.conv_scale is not None:
            scale_out = self.conv_scale(x)

        out = scale_out + identity_out
        if self.conv_kxk is not None:
            for rc in self.conv_kxk:
                out += rc(x)

        return self.act(self.se(out))

    def reparameterize(self):
        """Re-parameterize multi-branched architecture to plain CNN-like structure."""
        if self.reparam_conv is not None:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.in_chs,
            out_channels=self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            dilation=self.dilation,
            groups=self.groups,
            bias_attr=True,
        )
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()

        self.__delattr__("conv_kxk")
        self.__delattr__("conv_scale")
        if hasattr(self, "identity"):
            self.__delattr__("identity")

        self.inference_mode = True

    def _get_kernel_bias(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Obtain re-parameterized kernel and bias."""
        kernel_scale = 0
        bias_scale = 0
        if self.conv_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.conv_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(kernel_scale, [pad, pad, pad, pad])

        kernel_identity = 0
        bias_identity = 0
        if self.identity is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv = 0
        bias_conv = 0
        if self.conv_kxk is not None:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(self.conv_kxk[ix])
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(
            self,
            branch: Union[nn.Sequential, nn.BatchNorm2D]
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Fuse batchnorm layer with preceding conv layer."""
        if isinstance(branch, ConvNormAct):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_chs // self.groups
                kernel_value = np.zeros(
                    (self.in_chs, input_dim, self.kernel_size, self.kernel_size),
                    dtype=np.float32)
                for i in range(self.in_chs):
                    kernel_value[
                        i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2
                    ] = 1
                self.id_tensor = paddle.to_tensor(kernel_value)
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Layer):
    """Building Block of RepLKNet.

    This class defines overparameterized large kernel conv block.
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int,
            group_size: int,
            small_kernel: Optional[int] = None,
            use_se: bool = False,
            act_layer: Optional[nn.Layer] = None,
            inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.stride = stride
        self.groups = num_groups(group_size, in_chs)
        self.in_chs = in_chs
        self.out_chs = out_chs

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                dilation=1,
                groups=self.groups,
                bias_attr=True,
            )
        else:
            self.reparam_conv = None
            self.large_conv = ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=kernel_size // 2,
                groups=self.groups,
                apply_act=False,
            )
            if small_kernel is not None:
                assert small_kernel <= kernel_size, \
                    "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = ConvNormAct(
                    in_chs,
                    out_chs,
                    kernel_size=small_kernel,
                    stride=self.stride,
                    padding=small_kernel // 2,
                    groups=self.groups,
                    apply_act=False,
                )
        self.se = SqueezeExcite(out_chs, rd_ratio=0.25) if use_se else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.reparam_conv is not None:
            out = self.reparam_conv(x)
        else:
            out = self.large_conv(x)
            if self.small_conv is not None:
                out = out + self.small_conv(x)
        out = self.se(out)
        out = self.act(out)
        return out

    def get_kernel_bias(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Obtain re-parameterized kernel and bias."""
        eq_k, eq_b = self._fuse_bn(self.large_conv.conv, self.large_conv.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += F.pad(
                small_k, [(self.kernel_size - self.small_kernel) // 2] * 4
            )
        return eq_k, eq_b

    def reparameterize(self) -> None:
        """Re-parameterize multi-branched architecture to plain CNN-like structure."""
        eq_k, eq_b = self.get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            self.in_chs,
            self.out_chs,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.kernel_size // 2,
            groups=self.groups,
            bias_attr=True,
        )

        self.reparam_conv.weight.set_value(eq_k)
        self.reparam_conv.bias.set_value(eq_b)
        self.__delattr__("large_conv")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

    @staticmethod
    def _fuse_bn(
            conv: nn.Conv2D,
            bn: nn.BatchNorm2D
    ) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """Fuse batchnorm layer with conv layer."""
        kernel = conv.weight
        running_mean = bn._mean
        running_var = bn._variance
        gamma = bn.weight
        beta = bn.bias
        eps = bn._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


def convolutional_stem(
        in_chs: int,
        out_chs: int,
        act_layer: Type[nn.Layer] = nn.GELU,
        inference_mode: bool = False,
        use_scale_branch: bool = True,
) -> nn.Sequential:
    """Build convolutional stem with MobileOne blocks."""
    return nn.Sequential(
        MobileOneBlock(
            in_chs=in_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=3,
            stride=2,
            group_size=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
        ),
        MobileOneBlock(
            in_chs=out_chs,
            out_chs=out_chs,
            kernel_size=1,
            stride=1,
            act_layer=act_layer,
            inference_mode=inference_mode,
            use_scale_branch=use_scale_branch,
        ),
    )


class Attention(nn.Layer):
    """Multi-headed Self Attention module."""

    def __init__(
            self,
            dim: int,
            head_dim: int = 32,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.flatten(2).transpose([0, 2, 1])
        qkv = (
            self.qkv(x)
            .reshape([B, N, 3, self.num_heads, self.head_dim])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x


class PatchEmbed(nn.Layer):
    """Convolutional patch embedding layer."""

    def __init__(
            self,
            patch_size: int,
            stride: int,
            in_chs: int,
            embed_dim: int,
            act_layer: Type[nn.Layer] = nn.GELU,
            lkc_use_act: bool = False,
            use_se: bool = False,
            inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_chs=in_chs,
                out_chs=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                group_size=1,
                small_kernel=3,
                use_se=use_se,
                act_layer=act_layer if lkc_use_act else None,
                inference_mode=inference_mode,
            ),
            MobileOneBlock(
                in_chs=embed_dim,
                out_chs=embed_dim,
                kernel_size=1,
                stride=1,
                use_se=False,
                act_layer=act_layer,
                inference_mode=inference_mode,
            )
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        return x


class LayerScale2d(nn.Layer):
    def __init__(
            self,
            dim: int,
            init_values: float = 1e-5,
            inplace: bool = False,
    ):
        super().__init__()
        self.inplace = inplace
        self.gamma = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=nn.initializer.Constant(value=init_values)
        )

    def forward(self, x):
        return x * self.gamma


class RepMixer(nn.Layer):
    """Reparameterizable token mixer."""

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            layer_scale_init_value: Optional[float] = 1e-5,
            inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                self.dim,
                self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias_attr=True,
            )
        else:
            self.reparam_conv = None
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                group_size=1,
                use_act=False,
            )
            if layer_scale_init_value is not None:
                self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
            else:
                self.layer_scale = nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = x + self.layer_scale(self.mixer(x) - self.norm(x))
        return x

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single convolutional layer."""
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        if isinstance(self.layer_scale, LayerScale2d):
            w = self.mixer.id_tensor + self.layer_scale.gamma.unsqueeze(0) * (
                    self.mixer.reparam_conv.weight - self.norm.reparam_conv.weight
            )
            b = paddle.squeeze(self.layer_scale.gamma) * (
                    self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias
            )
        else:
            w = (
                    self.mixer.id_tensor
                    + self.mixer.reparam_conv.weight
                    - self.norm.reparam_conv.weight
            )
            b = self.mixer.reparam_conv.bias - self.norm.reparam_conv.bias

        self.reparam_conv = nn.Conv2D(
            self.dim,
            self.dim,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size // 2,
            groups=self.dim,
            bias_attr=True,
        )
        self.reparam_conv.weight.set_value(w)
        self.reparam_conv.bias.set_value(b)

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("mixer")
        self.__delattr__("norm")
        self.__delattr__("layer_scale")


class ConvMlp(nn.Layer):
    """Convolutional FFN Module."""

    def __init__(
            self,
            in_chs: int,
            hidden_channels: Optional[int] = None,
            out_chs: Optional[int] = None,
            act_layer: Type[nn.Layer] = nn.GELU,
            drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvNormAct(
            in_chs,
            out_chs,
            kernel_size=7,
            padding=3,
            groups=in_chs,
            apply_act=False,
        )
        self.fc1 = nn.Conv2D(in_chs, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_channels, out_chs, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepConditionalPosEnc(nn.Layer):
    """Implementation of conditional positional encoding."""

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
            inference_mode: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), (
            f'"spatial_shape" must by a sequence or int, '
            f"get {type(spatial_shape)} instead."
        )
        assert len(spatial_shape) == 2, (
            f'Length of "spatial_shape" should be 2, '
            f"got {len(spatial_shape)} instead."
        )

        self.spatial_shape = spatial_shape
        self.dim = dim
        self.dim_out = dim_out or dim
        self.groups = dim

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                self.dim,
                self.dim_out,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=spatial_shape[0] // 2,
                groups=self.groups,
                bias_attr=True,
            )
        else:
            self.reparam_conv = None
            self.pos_enc = nn.Conv2D(
                self.dim,
                self.dim_out,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                groups=self.groups,
                bias_attr=True,
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.reparam_conv is not None:
            x = self.reparam_conv(x)
        else:
            x = self.pos_enc(x) + x
        return x

    def reparameterize(self) -> None:
        input_dim = self.dim // self.groups
        kernel_value = np.zeros(
            (
                self.dim,
                input_dim,
                self.spatial_shape[0],
                self.spatial_shape[1],
            ),
            dtype=np.float32,
        )
        for i in range(self.dim):
            kernel_value[
                i,
                i % input_dim,
                self.spatial_shape[0] // 2,
                self.spatial_shape[1] // 2,
            ] = 1
        id_tensor = paddle.to_tensor(kernel_value)

        w_final = id_tensor + self.pos_enc.weight
        b_final = self.pos_enc.bias

        self.reparam_conv = nn.Conv2D(
            self.dim,
            self.dim_out,
            kernel_size=self.spatial_shape,
            stride=1,
            padding=int(self.spatial_shape[0] // 2),
            groups=self.groups,
            bias_attr=True,
        )
        self.reparam_conv.weight.set_value(w_final)
        self.reparam_conv.bias.set_value(b_final)

        for name, para in self.named_parameters():
            if 'reparam_conv' in name:
                continue
            para.detach_()
        self.__delattr__("pos_enc")


class RepMixerBlock(nn.Layer):
    """Implementation of Metaformer block with RepMixer as token mixer."""

    def __init__(
            self,
            dim: int,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            act_layer: Type[nn.Layer] = nn.GELU,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-5,
            inference_mode: bool = False,
    ):
        super().__init__()

        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )

        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = self.token_mixer(x)
        x = x + self.drop_path(self.layer_scale(self.mlp(x)))
        return x


class AttentionBlock(nn.Layer):
    """Implementation of metaformer block with MHSA as token mixer."""

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            act_layer: Type[nn.Layer] = nn.GELU,
            norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = Attention(dim=dim)
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale2d(dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.layer_scale_1(self.token_mixer(self.norm(x))))
        x = x + self.drop_path2(self.layer_scale_2(self.mlp(x)))
        return x


class FastVitStage(nn.Layer):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            depth: int,
            token_mixer_type: str,
            downsample: bool = True,
            se_downsample: bool = False,
            down_patch_size: int = 7,
            down_stride: int = 2,
            pos_emb_layer: Optional[nn.Layer] = None,
            kernel_size: int = 3,
            mlp_ratio: float = 4.0,
            act_layer: Type[nn.Layer] = nn.GELU,
            norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
            proj_drop_rate: float = 0.0,
            drop_path_rate: Union[List[float], float] = 0.0,
            layer_scale_init_value: Optional[float] = 1e-5,
            lkc_use_act: bool = False,
            inference_mode: bool = False,
    ):
        super().__init__()

        if downsample:
            self.downsample = PatchEmbed(
                patch_size=down_patch_size,
                stride=down_stride,
                in_chs=dim,
                embed_dim=dim_out,
                use_se=se_downsample,
                act_layer=act_layer,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
            )
        else:
            assert dim == dim_out
            self.downsample = nn.Identity()

        if pos_emb_layer is not None:
            self.pos_emb = pos_emb_layer(dim_out, inference_mode=inference_mode)
        else:
            self.pos_emb = nn.Identity()

        blocks = []
        for block_idx in range(depth):
            if token_mixer_type == "repmixer":
                blocks.append(RepMixerBlock(
                    dim_out,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                ))
            elif token_mixer_type == "attention":
                blocks.append(AttentionBlock(
                    dim_out,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    drop_path=drop_path_rate[block_idx],
                    layer_scale_init_value=layer_scale_init_value,
                ))
            else:
                raise ValueError(
                    "Token mixer type: {} not supported".format(token_mixer_type)
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.pos_emb(x)
        x = self.blocks(x)
        return x


class FastViT(nn.Layer):
    """
    This class implements FastViT architecture.
    """

    def __init__(
            self,
            in_chans: int = 3,
            layers: Tuple[int, ...] = (2, 2, 6, 2),
            token_mixers: Tuple[str, ...] = ("repmixer", "repmixer", "repmixer", "repmixer"),
            embed_dims: Tuple[int, ...] = (64, 128, 256, 512),
            mlp_ratios: Tuple[float, ...] = (4,) * 4,
            downsamples: Tuple[bool, ...] = (False, True, True, True),
            se_downsamples: Tuple[bool, ...] = (False, False, False, False),
            repmixer_kernel_size: int = 3,
            num_classes: int = 1000,
            pos_embs: Tuple[Optional[nn.Layer], ...] = (None,) * 4,
            down_patch_size: int = 7,
            down_stride: int = 2,
            drop_rate: float = 0.0,
            proj_drop_rate: float = 0.0,
            drop_path_rate: float = 0.0,
            layer_scale_init_value: float = 1e-5,
            lkc_use_act: bool = False,
            stem_use_scale_branch: bool = True,
            fork_feat: bool = False,
            cls_ratio: float = 2.0,
            global_pool: str = 'avg',
            norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
            act_layer: Type[nn.Layer] = nn.GELU,
            inference_mode: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = 0 if fork_feat else num_classes
        self.fork_feat = fork_feat
        self.global_pool = global_pool
        self.feature_info = []

        # Convolutional stem
        self.stem = convolutional_stem(
            in_chans,
            embed_dims[0],
            act_layer,
            inference_mode,
            use_scale_branch=stem_use_scale_branch,
        )

        # Build the main stages of the network architecture
        prev_dim = embed_dims[0]
        scale = 1
        dpr = self._calculate_drop_path_rates(drop_path_rate, layers)
        stages = []
        for i in range(len(layers)):
            downsample = downsamples[i] or prev_dim != embed_dims[i]
            stage = FastVitStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=layers[i],
                downsample=downsample,
                se_downsample=se_downsamples[i],
                down_patch_size=down_patch_size,
                down_stride=down_stride,
                pos_emb_layer=pos_embs[i],
                token_mixer_type=token_mixers[i],
                kernel_size=repmixer_kernel_size,
                mlp_ratio=mlp_ratios[i],
                act_layer=act_layer,
                norm_layer=norm_layer,
                proj_drop_rate=proj_drop_rate,
                drop_path_rate=dpr[i],
                layer_scale_init_value=layer_scale_init_value,
                lkc_use_act=lkc_use_act,
                inference_mode=inference_mode,
            )
            stages.append(stage)
            prev_dim = embed_dims[i]
            if downsample:
                scale *= 2
            self.feature_info += [dict(num_chs=prev_dim, reduction=4 * scale, module=f'stages.{i}')]
        self.stages = nn.Sequential(*stages)
        self.num_stages = len(self.stages)
        self.num_features = self.head_hidden_size = prev_dim

        # For segmentation and detection, extract intermediate output
        if self.fork_feat:
            self.out_indices = [0, 1, 2, 3]
            for i_emb, i_layer in enumerate(self.out_indices):
                layer = norm_layer(embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_sublayer(layer_name, layer)
        else:
            # Classifier head
            self.num_features = self.head_hidden_size = final_features = int(embed_dims[-1] * cls_ratio)
            self.final_conv = MobileOneBlock(
                in_chs=embed_dims[-1],
                out_chs=final_features,
                kernel_size=3,
                stride=1,
                group_size=1,
                inference_mode=inference_mode,
                use_se=True,
                act_layer=act_layer,
                num_conv_branches=1,
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Flatten(),
                nn.Dropout(drop_rate),
                nn.Linear(final_features, num_classes),
            )

    def _calculate_drop_path_rates(self, drop_path_rate, layers):
        """Calculate drop path rates for each stage and block."""
        total_depth = sum(layers)
        dpr = [drop_path_rate * i / (total_depth - 1) for i in range(total_depth)]
        dpr_iter = iter(dpr)
        dpr_per_stage = []
        for depth in layers:
            stage_dpr = [next(dpr_iter) for _ in range(depth)]
            dpr_per_stage.append(stage_dpr)
        return dpr_per_stage

    def forward_features(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.stem(x)
        outs = []
        for idx, block in enumerate(self.stages):
            x = block(x)
            if self.fork_feat:
                if idx in self.out_indices:
                    norm_layer = getattr(self, f"norm{idx}")
                    x_out = norm_layer(x)
                    outs.append(x_out)
        if self.fork_feat:
            return outs
        x = self.final_conv(x)
        return x

    def forward_head(self, x: paddle.Tensor, pre_logits: bool = False):
        if pre_logits:
            x = self.head[:-1](x)
        else:
            x = self.head(x)
        return x

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.forward_features(x)
        if self.fork_feat:
            return x
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


def FastViT_T8(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-T8 model variant."""
    model_args = dict(
        layers=(2, 2, 4, 2),
        embed_dims=(48, 96, 192, 384),
        mlp_ratios=(3, 3, 3, 3),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer")
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_T8", ""), use_ssld=use_ssld)
    return model


def FastViT_T12(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-T12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(3, 3, 3, 3),
        token_mixers=("repmixer", "repmixer", "repmixer", "repmixer"),
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_T12", ""), use_ssld=use_ssld)
    return model


def FastViT_SA12(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-SA12 model variant."""
    model_args = dict(
        layers=(2, 2, 6, 2),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_SA12", ""), use_ssld=use_ssld)
    return model


def FastViT_SA24(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-SA24 model variant."""
    model_args = dict(
        layers=(4, 4, 12, 4),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_SA24", ""), use_ssld=use_ssld)
    return model


def FastViT_SA36(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-SA36 model variant."""
    model_args = dict(
        layers=(6, 6, 18, 6),
        embed_dims=(64, 128, 256, 512),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention"),
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_SA36", ""), use_ssld=use_ssld)
    return model


def FastViT_MA36(pretrained=False, use_ssld=False, **kwargs):
    """Instantiate FastViT-MA36 model variant."""
    model_args = dict(
        layers=(6, 6, 18, 6),
        embed_dims=(76, 152, 304, 608),
        mlp_ratios=(4, 4, 4, 4),
        pos_embs=(None, None, None, partial(RepConditionalPosEnc, spatial_shape=(7, 7))),
        token_mixers=("repmixer", "repmixer", "repmixer", "attention")
    )
    model = FastViT(**dict(model_args, **kwargs))
    _load_pretrained(
        pretrained, model, MODEL_URLS.get("FastViT_MA36", ""), use_ssld=use_ssld)
    return model
