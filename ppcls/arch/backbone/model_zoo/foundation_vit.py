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
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "CLIP_small_patch16_224": None,
    "CLIP_base_patch32_224": None,
    "CLIP_base_patch16_224": None,
    "CLIP_large_patch14_336": None,
    "CLIP_large_patch14_224": None,
    "BEiTv2_base_patch16_224": None,
    "BEiTv2_large_patch16_224": None,
    "CAE_base_patch16_224": None,
    "EVA_small_patch16_224": None,
    "MOCOV3_small": None,
    "MOCOV3_base": None,
    "MAE_huge_patch14": None,
    "MAE_large_patch16": None,
    "MAE_base_patch16": None
}

__all__ = list(MODEL_URLS.keys())

_model_size = None
_model_diff = None

_CLIP_diff = {
    'add_layer_norm_before_encoder': [
        'base_patch32_224', 'base_patch16_224', 'large_patch14_336',
        'large_patch14_224'
    ],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [
        'base_patch32_224', 'base_patch16_224', 'large_patch14_336',
        'large_patch14_224'
    ],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
    }
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
    }
}

_CoCa_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': [],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': [],
    'remove_cls_token': ['small_patch16_224'],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
    }
}

_BEiTv2_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa':
    ['base_patch16_224', 'large_patch16_224'],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': ['base_patch16_224', 'large_patch16_224'],
    'remove_cls_token': [],
    'remove_abs_pos_emb': ['base_patch16_224', 'large_patch16_224'],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],
        'return_all_tokens': [],
        'return_patch_tokens': [],
    }
}

_CAE_diff = {
    'add_layer_norm_before_encoder': [],
    'add_relative_position_bias_in_msa': ['base_patch16_224'],
    'add_shared_rel_pos_bias': [],
    'add_mul_gamma_to_msa_mlp': ['base_patch16_224'],
    'remove_cls_token': [],
    'remove_abs_pos_emb': [],
    'replace_mlp_GELU': [],
    'head': {
        'fc_norm': [],  # 3 x 197 x 786
        'return_all_tokens': [],  # 3 x 197 x 1000
        'return_patch_tokens': [],  # 3 x 196 x 1000
    }
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
        'fc_norm': ['huge_patch14'],
        'return_all_tokens': [],
        'return_patch_tokens': [],
    }
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
        'fc_norm': ['huge_patch14'],
        'return_all_tokens': [],
        'return_patch_tokens': [],
    }
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
    keep_prob = paddle.to_tensor(1 - drop_prob)
    shape = (paddle.shape(x)[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape).astype(x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
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


class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * nn.functional.sigmoid(1.702 * x)


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
        self.act = act_layer() if _model_size not in _model_diff[
            'replace_mlp_GELU'] else QuickGELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 model_name=None,
                 window_size=None):
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

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

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
        # B= paddle.shape(x)[0]
        N, C = x.shape[1:]
        qkv = self.qkv(x).reshape((-1, N, 3, self.num_heads, C //
                                   self.num_heads)).transpose((2, 0, 3, 1, 4))
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
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
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
                 window_size=None):
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
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            model_name=self._model_name,
            window_size=window_size)
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
                       drop=drop)

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is not None:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class RelativePositionBias(nn.Layer):
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


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        x = self.proj(x).flatten(2).transpose((0, 2, 1))
        return x


class Head(nn.Layer):
    def __init__(self, embed_dim, class_num, norm_layer, model_size, setting):
        super().__init__()
        self.model_size = model_size
        self.setting = setting

        self.fc_norm = eval(norm_layer)(
            embed_dim,
            epsilon=1e-5) if model_size in setting['fc_norm'] else None
        self.return_all_tokens = model_size in setting['return_all_tokens']
        self.return_patch_tokens = model_size in setting['return_patch_tokens']

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
        else:
            if self.return_all_tokens:
                x = x
            elif self.return_patch_tokens:
                x = x[:, 1:]
            else:
                x = x[:, 0]
        return self.fc_head(x)


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch input
    """

    def __init__(self,
                 model_name,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-5,
                 **kwargs):
        super().__init__()
        global _model_diff
        global _model_size
        _model_split = model_name.split('_')
        self.model_name = _model_split[0]
        self.model_size = '_'.join(_model_split[1:])
        _model_size = self.model_size
        _model_diff = eval(f'_{self.model_name}_diff')

        self.class_num = class_num
        self.return_embed = kwargs.get('return_embed', True)
        self.num_features = self.embed_dim = embed_dim
        _img_size = to_2tuple(img_size)
        _patch_size = to_2tuple(patch_size)
        self.window_size = (_img_size[0] // _patch_size[0],
                            _img_size[1] // _patch_size[1])
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        if _model_size in _model_diff['add_shared_rel_pos_bias']:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.window_size, num_heads=num_heads)

        self.ln_pre = nn.LayerNorm(embed_dim) if _model_size in _model_diff[
            'add_layer_norm_before_encoder'] else nn.Identity()

        if _model_size in _model_diff['remove_cls_token']:
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

        if _model_size in _model_diff['remove_abs_pos_emb']:
            self.pos_embed = None
        else:
            self.add_parameter("pos_embed", self.pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)

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
                window_size=self.window_size) for i in range(depth)
        ])

        self.norm = eval(norm_layer)(embed_dim, epsilon=epsilon)

        self.head = Identity() if self.return_embed else Head(
            embed_dim, class_num, norm_layer, self.model_size,
            _model_diff['head'])

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed)
        if not _model_size in _model_diff['remove_cls_token']:
            trunc_normal_(self.cls_token)
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
        # B = x.shape[0]
        B = paddle.shape(x)[0]
        x = self.patch_embed(x)
        if not _model_size in _model_diff['remove_cls_token']:
            cls_tokens = self.cls_token.expand((B, -1, -1))
            x = paddle.concat((cls_tokens, x), axis=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed

        x = self.ln_pre(x)
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if hasattr(self,
                                                      'rel_pos_bias') else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def CLIP_base_patch32_224(pretrained=False, use_ssld=False, **kwargs):
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


def CLIP_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
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


def CLIP_large_patch14_336(pretrained=False, use_ssld=False, **kwargs):
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


def CLIP_large_patch14_224(pretrained=False, use_ssld=False, **kwargs):
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


def BEiTv2_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
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


def BEiTv2_large_patch16_224(pretrained=False, use_ssld=False, **kwargs):
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


def MOCOV3_small(pretrained=False, use_ssld=False, **kwargs):
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


def MOCOV3_base(pretrained=False, use_ssld=False, **kwargs):
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


def MAE_base_patch16(pretrained=False, use_ssld=False, **kwargs):
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


def MAE_large_patch16(pretrained=False, use_ssld=False, **kwargs):
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


def MAE_huge_patch14(pretrained=False, use_ssld=False, **kwargs):
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


def EVA_huge_patch14(pretrained=False, use_ssld=False, **kwargs):
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


def CAE_base_patch16_224(pretrained=False, use_ssld=False, **kwargs):
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