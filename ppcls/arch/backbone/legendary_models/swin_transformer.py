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

# Code was based on https://github.com/microsoft/Swin-Transformer
# reference: https://arxiv.org/abs/2103.14030

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

from ..model_zoo.vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity
from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "SwinTransformer_tiny_patch4_window7_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_small_patch4_window7_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_small_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_base_patch4_window7_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_base_patch4_window12_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_base_patch4_window12_384_pretrained.pdparams",
    "SwinTransformer_large_patch4_window7_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window7_224_pretrained.pdparams",
    "SwinTransformer_large_patch4_window12_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/SwinTransformer_large_patch4_window12_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

# The following re-implementation of roll is inspired by
# https://gitee.com/ascend/pytorch/blob/master/torch_npu/contrib/function/roll.py


def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)


class RollWithIndexSelect(paddle.autograd.PyLayer):
    @staticmethod
    def forward(ctx, input1, index_fp, index_bp):
        N, H, W, C = input1.shape
        ctx.input1 = input1
        ctx.index_bp = index_bp
        result = input1.reshape([N, H * W, C]).index_select(
            index_fp, 1).reshape([N, H, W, C])
        return result

    @staticmethod
    def backward(ctx, grad):
        input1 = ctx.input1
        N, H, W, C = input1.shape
        index_bp = ctx.index_bp
        grad_input = grad.reshape([N, H * W, C]).index_select(
            index_bp, 1).reshape([N, H, W, C])
        return grad_input, None, None


def get_roll_index(H, W, shifts, place):
    index = np.arange(0, H * W, dtype=np.int64).reshape([H, W])
    index_fp = np.roll(index, shift=shifts, axis=(0, 1)).reshape([-1])
    index_bp = {i: idx for idx, i in enumerate(index_fp.tolist())}
    index_bp = [index_bp[i] for i in range(H * W)]
    index_fp = paddle.to_tensor(index_fp, place=place)
    index_bp = paddle.to_tensor(index_fp, dtype='int64', place=place)
    return [index_fp, index_bp]


class NpuRollWithIndexSelect():
    def __init__(self):
        self.index_dict = {}
        self.roll_with_index_select = RollWithIndexSelect.apply

    def __call__(self, x, shifts, axis):
        assert x.dim() == 4
        assert len(shifts) == 2
        assert len(axis) == 2
        N, H, W, C = x.shape
        key = (H, W, shifts, axis)
        if key not in self.index_dict:
            self.index_dict[key] = get_roll_index(H, W, shifts, x.place)
        index_fp, index_bp = self.index_dict[key]
        return self.roll_with_index_select(x, index_fp, index_bp)


class RollWrapper(object):
    _roll = None

    @staticmethod
    def roll(x, shifts, axis):
        return RollWrapper._roll(x, shifts, axis)


# NOTE(xiongkun): paddle.SOT can't analysis this builtin function, which will cause subgraph break in sot.
# we do this here will not effect sot translate.
paddle_custom_device_types = paddle.device.get_all_custom_device_type()

if RollWrapper._roll is None:
    RollWrapper._roll = NpuRollWithIndexSelect(
    ) if 'npu' in paddle_custom_device_types else paddle.roll


class Mlp(nn.Layer):
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


def pading_for_not_divisible(pixel_values,
                             height,
                             width,
                             patch_size,
                             format="BCHW",
                             function="split"):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if function == "split":
        pading_width = patch_size[1] - width % patch_size[1]
        pading_height = patch_size[0] - height % patch_size[0]
    elif function == "merge":
        pading_width = width % 2
        pading_height = height % 2
    data_format = "NCHW"
    if format == "BCHW":
        zero_pad = paddle.to_tensor([0],dtype=paddle.int32)
        pad_index_h = paddle.concat([zero_pad,pading_height])
        pad_index_w = paddle.concat([zero_pad,pading_width])
        pad_index = paddle.concat([pad_index_h,pad_index_w])
    elif format == "BHWC":
        data_format = "NHWC"
        zero_pad = paddle.to_tensor([0],dtype=paddle.int32)
        pad_index_h = paddle.concat([zero_pad,pading_height])
        pad_index_w = paddle.concat([zero_pad,pading_width])
        pad_index = paddle.concat([pad_index_h,pad_index_w])
    else:
        assert ("vaild format")
    return F.pad(pixel_values, pad_index,data_format=data_format), pad_index


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.reshape(
        [B, H // window_size, window_size, W // window_size, window_size, C])
    windows = x.transpose([0, 1, 3, 2, 4, 5]).reshape(
        [-1, window_size, window_size, C])
    return windows


def window_reverse(windows, window_size, H, W, C):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    x = windows.reshape(
        [-1, H // window_size, W // window_size, window_size, window_size, C])
    x = x.transpose([0, 1, 3, 2, 4, 5]).reshape([-1, H, W, C])
    return x


class WindowAttention(nn.Layer):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self,
                 dim,
                 window_size,
                 num_heads,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # 2*Wh-1 * 2*Ww-1, nH
        self.relative_position_bias_table = self.create_parameter(
            shape=((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                   num_heads),
            default_initializer=zeros_)
        self.add_parameter("relative_position_bias_table",
                           self.relative_position_bias_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(self.window_size[0])
        coords_w = paddle.arange(self.window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww

        coords_flatten_1 = coords_flatten.unsqueeze(axis=2)
        coords_flatten_2 = coords_flatten.unsqueeze(axis=1)
        relative_coords = coords_flatten_1 - coords_flatten_2

        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[
            0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table)
        self.softmax = nn.Softmax(axis=-1)

    def eval(self, ):
        # this is used to re-param swin for model export
        relative_position_bias_table = self.relative_position_bias_table
        window_size = self.window_size
        index = self.relative_position_index.reshape([-1])

        relative_position_bias = paddle.index_select(
            relative_position_bias_table, index)
        relative_position_bias = relative_position_bias.reshape([
            window_size[0] * window_size[1], window_size[0] * window_size[1],
            -1
        ])  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.transpose(
            [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.unsqueeze(0)
        self.register_buffer("relative_position_bias", relative_position_bias)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(
            [B_, N, 3, self.num_heads, C // self.num_heads]).transpose(
                [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = paddle.mm(q, k.transpose([0, 1, 3, 2]))

        if self.training or not hasattr(self, "relative_position_bias"):
            index = self.relative_position_index.reshape([-1])

            relative_position_bias = paddle.index_select(
                self.relative_position_bias_table, index)
            relative_position_bias = relative_position_bias.reshape([
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1], -1
            ])  # Wh*Ww,Wh*Ww,nH

            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)
        else:
            attn = attn + self.relative_position_bias

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.reshape([B_ // nW, nW, self.num_heads, N, N
                                 ]) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.reshape([-1, self.num_heads, N, N])
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape([B_, N, C])
        x = paddle.mm(attn, v).transpose([0, 2, 1, 3]).reshape([B_, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self):
        return "dim={}, window_size={}, num_heads={}".format(
            self.dim, self.window_size, self.num_heads)

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Layer):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        H, W = self.input_resolution
        attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def get_attn_mask(self, height, width, dtype):
        if self.shift_size > 0:
            # calculate attention mask for shifted window multihead self attention
            img_mask = paddle.zeros((1, height, width, 1), dtype=dtype)
            height_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None), )
            width_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None), )
            count = 0
            for height_slice in height_slices:
                for width_slice in width_slices:
                    img_mask[:, height_slice, width_slice, :] = count
                    count += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.reshape(
                (-1, self.window_size * self.window_size))
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = masked_fill(attn_mask, attn_mask != 0, float(-100.0))
            attn_mask = masked_fill(attn_mask, attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        return attn_mask

    def forward(self, x, input_dimensions):
        B, H, W = input_dimensions
        _, L, C = paddle.shape(x)

        print(x)
        shortcut = x
        x = self.norm1(x)
        x = paddle.reshape(x,[B, H, W, C])
        x, pad_values = pading_for_not_divisible(x, H, W, self.window_size,
                                                 "BHWC")
        _, height_pad, width_pad, _ = paddle.shape(x)

        padding_state = pad_values[1] > 0 or pad_values[
            3] > 0  # change variable name
        # cyclic shift
        if self.shift_size > 0:
            shifted_x = RollWrapper.roll(
                x, shifts=(-self.shift_size, -self.shift_size), axis=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.reshape(
            [-1, self.window_size * self.window_size,
             C])  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        #check did it need to calculate again
        attn_mask = self.get_attn_mask(height_pad, width_pad, x.dtype)

        attn_windows = self.attn(
            x_windows, mask=attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.reshape(
            [-1, self.window_size, self.window_size, C])
        shifted_x = window_reverse(attn_windows, self.window_size, height_pad,
                                   width_pad, C)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = RollWrapper.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                axis=(1, 2))
        else:
            x = shifted_x

        if padding_state:
            x = x[:, :H, :W, :]
        x = paddle.reshape(x,[B, H * W, C])

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self):
        return "dim={}, input_resolution={}, num_heads={}, window_size={}, shift_size={}, mlp_ratio={}".format(
            self.dim, self.input_resolution, self.num_heads, self.window_size,
            self.shift_size, self.mlp_ratio)

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, input_dimensions):
        """
        x: B, H*W, C
        """
        B, H, W = input_dimensions
        B_, L, C = paddle.shape(x)[0], paddle.shape(x)[1], paddle.shape(x)[2]
        x = paddle.reshape(x,[B, H, W, C])
        print(x.shape)
        x, _ = pading_for_not_divisible(x, H, W, 2, "BHWC", function="merge")

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = paddle.concat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        # x = x.reshape([B, H // 2, 2, W // 2, 2, C])
        # x = x.transpose((0, 1, 3, 4, 2, 5))

        x = paddle.reshape(x,[B_, -1, self.dim*4])  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return "input_resolution={}, dim={}".format(self.input_resolution,
                                                    self.dim)

    def flops(self):
        B, H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Layer):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Layer, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Layer | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, input_dimensions):
        B, H, W = input_dimensions
        for blk in self.blocks:
            x = blk(x, input_dimensions)
        if self.downsample is not None:
            H, W = (H + 1) // 2, (W + 1) // 2
            x = self.downsample(x, input_dimensions)
        return x, (B, H, W)

    def extra_repr(self):
        return "dim={}, input_resolution={}, depth={}".format(
            self.dim, self.input_resolution, self.depth)

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Layer, optional): Normalization layer. Default: None
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = paddle.shape(x)
        x, _ = pading_for_not_divisible(x, H, W, self.patch_size, "BCHW")
        x = self.proj(x)
        B_, _, height, width = paddle.shape(x)
        output_dimensions = (B_, height, width)
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, output_dimensions

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (
            self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class SwinTransformer(TheseusLayer):
    """ Swin Transformer
        A PaddlePaddle impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Layer): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 **kwargs):
        super(SwinTransformer, self).__init__()

        self.num_classes = num_classes = class_num
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim), default_initializer=zeros_)
            self.add_parameter("absolute_pos_embed", self.absolute_pos_embed)
            trunc_normal_(self.absolute_pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule

        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(patches_resolution[0] // (2**i_layer),
                                  patches_resolution[1] // (2**i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging
                if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1D(1)
        self.head = nn.Linear(
            self.num_features,
            num_classes) if self.num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x, output_dimensions = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x, output_dimensions = layer(x, output_dimensions)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose([0, 2, 1]))  # B C 1
        x = paddle.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for _, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[
            0] * self.patches_resolution[1] // (2**self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


def _load_pretrained(pretrained,
                     model,
                     model_url,
                     use_ssld=False,
                     use_imagenet22k_pretrained=False,
                     use_imagenet22kto1k_pretrained=False,
                     **kwargs):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(
            model,
            model_url,
            use_ssld=use_ssld,
            use_imagenet22k_pretrained=use_imagenet22k_pretrained,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def SwinTransformer_tiny_patch4_window7_224(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=False,
        **kwargs):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.1
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_tiny_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformer_small_patch4_window7_224(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=False,
        **kwargs):
    model = SwinTransformer(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_small_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformer_base_patch4_window7_224(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=False,
        **kwargs):
    model = SwinTransformer(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_base_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformer_base_patch4_window12_384(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=False,
        **kwargs):
    model = SwinTransformer(
        img_size=384,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.5,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_base_patch4_window12_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformer_large_patch4_window7_224(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=True,
        **kwargs):
    model = SwinTransformer(
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_large_patch4_window7_224"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def SwinTransformer_large_patch4_window12_384(
        pretrained=False,
        use_ssld=False,
        use_imagenet22k_pretrained=False,
        use_imagenet22kto1k_pretrained=True,
        **kwargs):
    model = SwinTransformer(
        img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["SwinTransformer_large_patch4_window12_384"],
        use_ssld=use_ssld,
        use_imagenet22k_pretrained=use_imagenet22k_pretrained,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model
