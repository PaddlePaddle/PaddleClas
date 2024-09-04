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

# Code was based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# reference: https://arxiv.org/abs/2010.11929

from collections.abc import Callable, Iterable

import numpy as np
import paddle
import paddle.nn as nn
import sys
from paddle.nn.initializer import TruncatedNormal, Constant, Normal, Assign

from ....utils import logger
from ....utils.save_load import load_dygraph_pretrain
from ..base.theseus_layer import TheseusLayer

MODEL_URLS = {
    "CLIP_vit_base_patch32_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_base_patch32_224.pdparams",
    "CLIP_vit_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_base_patch16_224.pdparams",
    "CLIP_vit_large_patch14_336":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_large_patch14_336.pdparams",
    "CLIP_vit_large_patch14_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CLIP_vit_large_patch14_224.pdparams",
    "BEiTv2_vit_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_base_patch16_224.pdparams",
    "BEiTv2_vit_large_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/BEiTv2_vit_large_patch16_224.pdparams",
    "CAE_vit_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/CAE_vit_base_patch16_224.pdparams",
    'EVA_vit_giant_patch14':
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/EVA_vit_giant_patch14.pdparams",
    "MOCOV3_vit_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MOCOV3_vit_small.pdparams",
    "MOCOV3_vit_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MOCOV3_vit_base.pdparams",
    "MAE_vit_huge_patch14":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_huge_patch14.pdparams",
    "MAE_vit_large_patch16":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_large_patch16.pdparams",
    "MAE_vit_base_patch16":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/foundation_models/MAE_vit_base_patch16.pdparams",
}


def check_support_fused_op(use_fused_linear):
    if use_fused_linear:
        if paddle.device.cuda.get_device_capability()[0] >= 8:
            return True
        else:
            logger.warning("The current device don't support Fused OP! Using the general Linear instead.")
    return False


def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (torch.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        torch.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape([-1, src_h, src_w, C]).transpose(
        [0, 3, 1, 2])

    # The cubic interpolate algorithm only accepts float32
    dst_weight = paddle.nn.functional.interpolate(
        paddle.cast(src_weight, paddle.float32),
        size=dst_shape,
        align_corners=False,
        mode=mode)
    dst_weight = paddle.flatten(dst_weight, 2).transpose([0, 2, 1])
    dst_weight = paddle.cast(dst_weight, src_weight.dtype)

    return paddle.concat((extra_tokens, dst_weight), axis=1)


def pading_for_not_divisible(pixel_values,
                             height,
                             width,
                             patch_size,
                             format="BCHW",
                             function="split"):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if height % patch_size[0] == 0 and width % patch_size[1] == 0:
        return pixel_values, None
    if function == "split":
        pading_width = patch_size[1] - width % patch_size[1]
        pading_height = patch_size[0] - height % patch_size[0]
    elif function == "merge":
        pading_width = width % 2
        pading_height = height % 2
    if format == "BCHW":
        pad_index = (0, 0, 0, 0, 0, pading_height, 0, pading_width)
    elif format == "BHWC":
        pad_index = (0, 0, 0, pading_height, 0, pading_width, 0, 0)
    else:
        assert ("vaild format")

    return paddle.nn.functional.pad(pixel_values, pad_index), pad_index


__all__ = list(MODEL_URLS.keys())

_model_size = None
_model_diff = None

_CLIP_diff = {
    'add_layer_norm_before_encoder': [
        'vit_base_patch32_224', 'vit_base_patch16_224',
        'vit_large_patch14_336', 'vit_large_patch14_224'
    ],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': ['vit_base_patch16_224'],
    },
    'remove_cls_token_in_forward': ['vit_base_patch16_224'],
}


_MOCOV3_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

_CoCa_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

_BEiTv2_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa':
    ['vit_base_patch16_224', 'vit_large_patch16_224'],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp':
    ['vit_base_patch16_224', 'vit_large_patch16_224'],
    'remove_cls_token': [],
    'remove_abs_pos_emb': ['vit_base_patch16_224', 'vit_large_patch16_224'],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

_CAE_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': ['vit_base_patch16_224'],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': ['vit_base_patch16_224'],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],  # 3 x 197 x 786
        'return_all_tokens': [],  # 3 x 197 x 1000
        'return_patch_tokens': [],  # 3 x 196 x 1000
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

_EVA_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': ['vit_huge_patch14'],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

_MAE_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': ['vit_huge_patch14'],
        'return_all_tokens': [],
        'return_patch_tokens': [],
        'return_tokens_mean': [],
    },
    'remove_cls_token_in_forward': [],
}

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def to_2tuple(x):
    return tuple([x] * 2)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(TheseusLayer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class QuickGELU(TheseusLayer):
    def forward(self, x):
        return x * nn.functional.sigmoid(1.702 * x)


class Mlp(TheseusLayer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 Linear=nn.Linear):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer() if _model_size not in _model_diff[
            'replace_mlp_GELU'] else QuickGELU()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(TheseusLayer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 model_name=None,
                 window_size=None,
                 use_fused_attn=False,
                 Linear=nn.Linear):
        super().__init__()
        self._model_name = model_name

        if _model_size in _model_diff['add_relative_position_bias_in_msa']:
            assert isinstance(
                window_size, Iterable
            ), f'window_size must be iterable, should not be {type(window_size)}'
            self.window_size = window_size
            self._register_relative_position_index(
                window_size=window_size,
                num_heads=num_heads, )

        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.use_fused_attn = use_fused_attn
        # TODO: support mask
        if use_fused_attn:
            if hasattr(self, 'relative_position_bias_table') or (_model_size in _model_diff['add_shared_rel_pos_bias'] and rel_pos_bias is not None):
                logger.warning("The fused attn don't support `relative_position` yet, so fused attn will not be used.")
                self.use_fused_attn = False

    def _register_relative_position_index(
            self,
            window_size,
            num_heads, ):
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads],
            default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros((window_size[0] * window_size[1] + 1, ) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(
            -1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                             relative_position_index)

    def forward(self, x, rel_pos_bias=None):
        # B= x.shape[0]
        N, C = x.shape[1], x.shape[2]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C // self.num_heads))

        if not self.use_fused_attn:
            qkv = qkv.transpose((2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
            if hasattr(self, 'relative_position_bias_table'):
                relative_position_bias = \
                    self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                        self.window_size[0] * self.window_size[1] + 1,
                        self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
                relative_position_bias = relative_position_bias.transpose(
                    [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
                attn = attn + relative_position_bias.unsqueeze(0)

            if _model_size in _model_diff[
                    'add_shared_rel_pos_bias'] and rel_pos_bias is not None:
                attn = attn + rel_pos_bias

            attn = nn.functional.softmax(attn, axis=-1)
            attn = self.attn_drop(attn).matmul(v)
            attn = attn.transpose((0, 2, 1, 3))
        else:
            qkv = qkv.transpose((2, 0, 1, 3, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            # TODO: support mask
            attn = paddle.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)

        x = attn.reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(TheseusLayer):
    def __init__(self,
                 dim,
                 num_heads,
                 model_name,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 init_values=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 window_size=None,
                 use_fused_attn=False,
                 use_fused_linear=False):
        super().__init__()
        global _model_size
        global _model_diff
        self._model_name = model_name
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm1 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        Linear = paddle.incubate.nn.FusedLinear if use_fused_linear else nn.Linear
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            model_name=self._model_name,
            window_size=window_size,
            use_fused_attn=use_fused_attn,
            Linear=Linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

        if _model_size in _model_diff['add_mul_gamma_to_msa_mlp']:
            self.gamma_1 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
        else:
            self.gamma_1 = None
            self.gamma_2 = None

        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        elif isinstance(norm_layer, Callable):
            self.norm2 = norm_layer(dim)
        else:
            raise TypeError(
                "The norm_layer must be str or paddle.nn.layer.Layer class")
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       Linear=Linear)

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            atten_result = self.drop_path(
                self.attn(
                    self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + atten_result
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(TheseusLayer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads],
            default_initializer=zeros_)  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = paddle.arange(window_size[0])
        coords_w = paddle.arange(window_size[1])
        coords = paddle.stack(paddle.meshgrid(
            [coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = paddle.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :,
                                         None] - coords_flatten[:,
                                                                None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.transpose(
            [1, 2, 0])  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            paddle.zeros((window_size[0] * window_size[1] + 1,) * 2, dtype=relative_coords.dtype)
        relative_position_index[1:, 1:] = relative_coords.sum(
            -1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0:] = self.num_relative_distance - 3
        relative_position_index[0:, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer("relative_position_index",
                             relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww


class PatchEmbed(TheseusLayer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 conv_bias=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        if conv_bias:
            self.proj = nn.Conv2D(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2D(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                bias_attr=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x, _ = pading_for_not_divisible(x, H, W, patch_size=self.patch_size)

        x = self.proj(x)
        _, _, H, W = x.shape

        x = x.flatten(2).transpose((0, 2, 1))
        return x, (H, W)


class Head(TheseusLayer):
    def __init__(self, embed_dim, class_num, norm_layer, model_size, setting):
        super().__init__()
        self.model_size = model_size
        self.setting = setting

        self.fc_norm = eval(norm_layer)(
            embed_dim,
            epsilon=1e-5) if model_size in setting['fc_norm'] else None
        self.return_all_tokens = model_size in setting['return_all_tokens']
        self.return_patch_tokens = model_size in setting['return_patch_tokens']
        self.return_tokens_mean = model_size in setting['return_tokens_mean']

        self.fc_head = nn.Linear(embed_dim,
                                 class_num) if class_num > 0 else Identity()

    def forward(self, x):
        if self.fc_norm is not None:
            if self.return_all_tokens:
                x = self.fc_norm(x)
            else:
                t = x[:, 1:]
                if self.return_patch_tokens:
                    x = self.fc_norm(t)
                else:
                    x = self.fc_norm(t.mean(1))
        elif isinstance(self.fc_head, Identity):
            if self.return_all_tokens:
                x = x
            elif self.return_patch_tokens:
                x = x[:, 1:]
            elif self.return_tokens_mean:
                x = x.mean(1)
            else:
                x = x[:, 0]
        else:
            x = x
        return self.fc_head(x)


class VisionTransformer(TheseusLayer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 model_name,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 output_dim=512,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 image_project=False,
                 conv_bias=False,
                 feature_frame=False,
                 hugging_face_framework=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 head_init_scale=0.001,
                 **kwargs):
        super().__init__()
        global _model_diff
        global _model_size
        _model_split = model_name.split('_')
        self.model_name = _model_split[0]
        self.feature_frame = feature_frame
        self.model_size = '_'.join(_model_split[1:])
        _model_size = self.model_size
        _model_diff = eval(f'_{self.model_name}_diff')

        self.class_num = class_num
        self.return_embed = kwargs.get('return_embed', False)
        self.return_mean_embed = kwargs.get('return_mean_embed', False) and self.return_embed
        self.num_features = self.embed_dim = embed_dim
        use_fused_attn = check_support_fused_op(kwargs.get('use_fused_attn', False))
        use_fused_linear = check_support_fused_op(kwargs.get('use_fused_linear', False))
        _img_size = to_2tuple(img_size)
        _patch_size = to_2tuple(patch_size)
        self.window_size = (_img_size[0] // _patch_size[0],
                            _img_size[1] // _patch_size[1])
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            conv_bias=conv_bias)
        num_patches = self.patch_embed.num_patches

        if _model_size in _model_diff['add_shared_rel_pos_bias']:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.window_size, num_heads=num_heads)

        #self.ln_pre = nn.LayerNorm(embed_dim) if _model_size in _model_diff[
        #    'add_layer_norm_before_encoder'] else nn.Identity()

        if _model_size in _model_diff['remove_cls_token'] or self.feature_frame:
            self.pos_embed = self.create_parameter(
                shape=(1, num_patches, embed_dim), default_initializer=zeros_)
            self.cls_token = None
        else:
            self.pos_embed = self.create_parameter(
                shape=(1, num_patches + 1, embed_dim),
                default_initializer=zeros_)
            self.cls_token = self.create_parameter(
                shape=(1, 1, embed_dim), default_initializer=zeros_)
            self.add_parameter("cls_token", self.cls_token)

        self.pos_drop = nn.Dropout(p=drop_rate)
        # for LaClip
        if image_project:
            image_projection = self.create_parameter(
                shape=(img_size, embed_dim),
                default_initializer=Assign(
                    paddle.empty((img_size, embed_dim))))
            self.add_parameter("image_projection", image_projection)
        else:
            self.image_projection = None
        self.hugging_face_framework = hugging_face_framework
        #for path size hugging face plan
        if hugging_face_framework:
            self.ln_pre = nn.LayerNorm(embed_dim)
            self.add_parameter("pos_embed", self.pos_embed)
        else:
            self.ln_pre = nn.Identity() if _model_size not in _model_diff[
                'add_layer_norm_before_encoder'] else nn.LayerNorm(embed_dim)
            if _model_size in _model_diff['remove_abs_pos_emb']:
                self.pos_embed = None
            else:
                self.add_parameter("pos_embed", self.pos_embed)

        #proj
        proj = self.create_parameter(
            shape=(embed_dim, ),
            default_initializer=Assign((embed_dim**-0.5) * paddle.randn((
                (embed_dim, output_dim)))))
        self.add_parameter("proj", proj)

        dpr = np.linspace(0, drop_path_rate, depth)

        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                model_name=self.model_name,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                window_size=self.window_size,
                use_fused_attn=use_fused_attn,
                use_fused_linear=use_fused_linear) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        self.head = Identity() if self.return_embed else Head(
            embed_dim, class_num, norm_layer, self.model_size,
            _model_diff['head'])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)
        if not _model_size in _model_diff['remove_cls_token'] and self.feature_frame == False:
            trunc_normal_(self.cls_token)

        self.apply(self._init_weights)

        if feature_frame:
            self.feature = nn.Sequential(
            nn.Linear(embed_dim * self.patch_embed.num_patches, embed_dim,bias_attr=False),
            nn.BatchNorm1D(embed_dim, epsilon=2e-5),
            nn.Linear(embed_dim, output_dim, bias_attr=False),
            nn.BatchNorm1D(output_dim, epsilon=2e-5))
            self.pos_drop = Identity()
            self.cls_token = None
            self.image_projection = Identity()
            self.proj = None
            self.ln_pre = Identity()

        if head_init_scale != 1:
            if not self.return_embed and class_num > 0:
                self.head.fc_head.weight.set_value(
                    self.head.fc_head.weight *
                    paddle.to_tensor(head_init_scale))
                self.head.fc_head.bias.set_value(
                    self.head.fc_head.bias * paddle.to_tensor(head_init_scale))
            else:
                logger.warning(
                    "Because the head or head.fc_head of ViT is Identity() class, the argument head_init_scale is invalid."
                )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x, output_dimensions = self.patch_embed(x)
        if not _model_size in _model_diff['remove_cls_token'] and (self.feature_frame==False):
            cls_tokens = self.cls_token.expand((B, -1, -1))
            x = paddle.concat((cls_tokens, x), axis=1)

        if self.pos_embed is not None:
            x = x + resize_pos_embed(self.pos_embed, self.window_size,
                                     output_dimensions)

        x = self.ln_pre(x)
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if hasattr(self,
                                                      'rel_pos_bias') else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        if _model_size in _model_diff['remove_cls_token_in_forward']:
            x = x[:, 1:, :]
        if self.hugging_face_framework or self.return_embed == False:
            pooled, token = x[:, 0], x[:, 1:]
        else:
            pooled = x
        x = self.norm(pooled)
        return x

    def forward(self, x):
        x = self.forward_features(x)

        if self.feature_frame:
            B, L, C = x.shape
            x = paddle.reshape(x,[B, -1])
            x = self.feature(x)

        x = self.head(x)

        if self.proj is not None and isinstance(self.head,Identity):
           x = x @self.proj
        if self.return_mean_embed:
            x = x.mean(1)
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


def CLIP_vit_base_patch32_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-5,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def CLIP_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-5,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def React_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "React_vit_base_patch16_224"
    model = VisionTransformer(
        model_name=model_name.replace("React","CLIP"),
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        hugging_face_framework=True,
        epsilon=1e-5,
        **kwargs, )
    return model


def React_vit_base_patch32_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "React_vit_base_patch32_224"
    model = VisionTransformer(
        model_name=model_name.replace("React","CLIP"),
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        hugging_face_framework=True,
        epsilon=1e-5,
        **kwargs, )
    return model


def LaCLIP_vit_base_patch32_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "LaCLIP_vit_base_patch32_224"
    model = VisionTransformer(
        model_name=model_name.replace("LaCLIP","CLIP"),
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        hugging_face_framework=True,
        epsilon=1e-5,
        **kwargs, )

    return model


def LaCLIP_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "LaCLIP_vit_base_patch16_224"
    model = VisionTransformer(
        model_name=model_name.replace("LaCLIP","CLIP"),
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        hugging_face_framework=True,
        epsilon=1e-5,
        **kwargs, )

    return model


def Unicom_vit_base_patch32_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "Unicom_vit_base_patch32_224"
    model = VisionTransformer(
        model_name=model_name.replace("Unicom","CLIP"),
        img_size=224,
        patch_size=32,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        conv_bias=True,
        feature_frame=True,
        hugging_face_framework=False,
        image_project=False,
        epsilon=1e-5,
        **kwargs, )

    return model


def Unicom_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = "Unicom_vit_base_patch16_224"
    model = VisionTransformer(
        model_name=model_name.replace("Unicom","CLIP"),
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=False,
        hugging_face_framework=False,
        image_project=False,
        feature_frame=True,
        conv_bias=True,
        epsilon=1e-5,
        **kwargs, )

    return model


def CLIP_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-5,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def CLIP_vit_large_patch14_336(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=336,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-5,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def CLIP_vit_large_patch14_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-5,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def BEiTv2_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def BEiTv2_vit_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def MOCOV3_vit_small(pretrained=False, use_ssld=False, **kwargs):
    """
    vit small in mocov3
    """
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def MOCOV3_vit_base(pretrained=False, use_ssld=False, **kwargs):
    """
    vit base in mocov3
    """
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def MAE_vit_base_patch16(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def MAE_vit_large_patch16(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def MAE_vit_huge_patch14(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def EVA_vit_giant_patch14(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        patch_size=14,
        embed_dim=1408,
        depth=40,
        num_heads=16,
        init_values=None,
        mlp_ratio=4.3637,
        qkv_bias=True,
        class_num=0,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model


def CAE_vit_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
    model_name = sys._getframe().f_code.co_name
    model = VisionTransformer(
        model_name=model_name,
        img_size=224,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        epsilon=1e-6,
        **kwargs, )
    _load_pretrained(
        pretrained, model, MODEL_URLS[model_name], use_ssld=use_ssld)
    return model
