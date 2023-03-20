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

# Code was heavily based on https://github.com/PaddlePaddle/VIMER/blob/main/CAE/models/modeling_finetune.py
# reference: https://arxiv.org/abs/2202.03026

import collections
from itertools import repeat
import math
import numpy as np
from functools import partial

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.download import get_weights_path_from_url

MODEL_URLS = {
    "cae_base_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/cae_base_patch16_224_pretrained.pdparams",
    "cae_large_patch16_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/cae_large_patch16_224_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


def trunc_normal_(tensor, mean=0., std=1.):
    nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)


def drop_path(x, drop_prob: float=0., training: bool=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor.floor_()  # binarize
    output = x / keep_prob * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)


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
        self.fc1 = nn.Linear(in_features, hidden_features, bias_attr=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias_attr=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
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
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()

        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.zeros_ = nn.initializer.Constant(value=0.)

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias_attr=False)
        if qkv_bias:
            self.q_bias = self.create_parameter(
                [all_head_dim], default_initializer=self.zeros_)
            self.v_bias = self.create_parameter(
                [all_head_dim], default_initializer=self.zeros_)
        else:
            self.q_bias = None
            self.v_bias = None

        if window_size:
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (
                2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = self.create_parameter(
                [self.num_relative_distance, num_heads],
                default_initializer=self.zeros_)  # 2*Wh-1 * 2*Ww-1, nH
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
            relative_coords[:, :, 0] += window_size[
                0] - 1  # shift to start from 0
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
        else:
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, bias_attr=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rel_pos_bias=None):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            k_bias = paddle.zeros_like(self.v_bias)
            k_bias.stop_gradient = True
            qkv_bias = paddle.concat((self.q_bias, k_bias, self.v_bias))
        # qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads]).transpose([2, 0, 3, 1, 4])
        qkv = F.linear(x=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape([B, N, 3, self.num_heads, -1]).transpose(
            [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @k.transpose([0, 1, 3, 2]))

        if self.relative_position_bias_table is not None:
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.transpose(
                [2, 0, 1])  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if rel_pos_bias is not None:
            attn = attn + rel_pos_bias

        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, -1])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 init_values=None,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 window_size=None,
                 attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            attn_head_dim=attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        if init_values > 0:
            self.gamma_1 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
            self.gamma_2 = self.create_parameter(
                [dim],
                default_initializer=nn.initializer.Constant(value=init_values))
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None):
        if self.gamma_1 is None:
            x = x + self.drop_path(
                self.attn(
                    self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.gamma_1 * self.attn(
                self.norm1(x), rel_pos_bias=rel_pos_bias))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        to_2tuple = _ntuple(2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0],
                            img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.out_chans = embed_dim
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias_attr=True)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose([0, 2, 1])
        return x

    def _init_weights(self):
        fan_out = self.out_chans
        fan_in = self.patch_size[0] * self.patch_size[1] * self.in_chans
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.XavierUniform(fan_in, fan_out))  # MAE
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return weight_attr, bias_attr


class RelativePositionBias(nn.Layer):
    def __init__(self, window_size, num_heads):
        super().__init__()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (
            2 * window_size[1] - 1) + 3
        self.zeros_ = nn.initializer.Constant(value=0.)
        self.relative_position_bias_table = self.create_parameter(
            [self.num_relative_distance, num_heads],
            default_initializer=self.zeros_)  # 2*Wh-1 * 2*Ww-1, nH
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

    def forward(self):
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.reshape([-1])].reshape([
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1])  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.transpose([2, 0, 1])  # nH, Wh*Ww, Wh*Ww


def get_sinusoid_encoding_table(n_position, d_hid, token=False):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if token:
        sinusoid_table = np.concatenate(
            [sinusoid_table, np.zeros([1, d_hid])], dim=0)

    return paddle.to_tensor(sinusoid_table).unsqueeze(0)


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 init_values=None,
                 use_abs_pos_emb=True,
                 use_rel_pos_bias=False,
                 use_shared_rel_pos_bias=False,
                 use_mean_pooling=True,
                 init_scale=0.001,
                 lin_probe=False,
                 sin_pos_emb=True,
                 args=None):
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.use_mean_pooling = use_mean_pooling

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.zeros_ = nn.initializer.Constant(value=0.)
        self.ones_ = nn.initializer.Constant(value=1.)

        self.cls_token = self.create_parameter(
            [1, 1, embed_dim], default_initializer=self.zeros_)

        self.use_abs_pos_emb = use_abs_pos_emb
        if use_abs_pos_emb:
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim],
                default_initializer=self.zeros_)
        elif sin_pos_emb:
            # sine-cosine positional embeddings is on the way
            self.pos_embed = self.create_parameter(
                [1, num_patches + 1, embed_dim],
                default_initializer=self.zeros_)
            self.pos_embed.set_value(
                self.build_2d_sincos_position_embedding(embed_dim))
            self.pos_embed.stop_gradient = True  # fixed sin-cos embedding
        else:
            self.pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(
                window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.LayerList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
                window_size=self.patch_embed.patch_shape
                if use_rel_pos_bias else None) for i in range(depth)
        ])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(
            embed_dim)

        self.lin_probe = lin_probe
        # NOTE: batch norm
        if lin_probe:
            # TODO
            from models.lincls_bn import LP_BatchNorm
            self.fc_norm = LP_BatchNorm(embed_dim, affine=False)
        else:
            if use_mean_pooling:
                self.fc_norm = norm_layer(embed_dim)
            else:
                self.fc_norm = None
        self.head = nn.Linear(embed_dim,
                              class_num) if class_num > 0 else nn.Identity()

        if self.pos_embed is not None and use_abs_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        self.head.weight.set_value(self.head.weight * init_scale)
        self.head.bias.set_value(self.head.bias * init_scale)

    def build_2d_sincos_position_embedding(self,
                                           embed_dim=768,
                                           temperature=10000.):
        h, w = self.patch_embed.patch_shape
        grid_w = paddle.arange(w, dtype=paddle.float32)
        grid_h = paddle.arange(h, dtype=paddle.float32)
        grid_w, grid_h = paddle.meshgrid(grid_w, grid_h)
        assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = paddle.arange(pos_dim, dtype=paddle.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = paddle.einsum('m,d->md', grid_w.flatten(), omega)
        out_h = paddle.einsum('m,d->md', grid_h.flatten(), omega)
        pos_emb = paddle.concat(
            [
                paddle.sin(out_w), paddle.cos(out_w), paddle.sin(out_h),
                paddle.cos(out_h)
            ],
            axis=1)[None, :, :]

        # if not self.use_mean_pooling:
        pe_token = paddle.zeros([1, 1, embed_dim], dtype=paddle.float32)
        pos_emb = paddle.concat([pe_token, pos_emb], axis=1)
        return pos_emb

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.set_value(param / math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight, layer_id + 1)
            rescale(layer.mlp.fc2.weight, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                self.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            self.zeros_(m.bias)
            self.ones_(m.weight)

    def get_num_layers(self):
        return len(self.blocks)

    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, class_num, global_pool=''):
        self.class_num = class_num
        self.head = nn.Linear(self.embed_dim,
                              class_num) if class_num > 0 else nn.Identity()

    def forward_features(self, x, is_train=True):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.shape

        cls_tokens = self.cls_token.expand([
            batch_size, -1, -1
        ]).astype(x.dtype)  # stole cls_tokens impl from Phil Wang, thanks
        x = paddle.concat((cls_tokens, x), axis=1)
        if self.pos_embed is not None:
            if self.use_abs_pos_emb:
                x = x + self.pos_embed.expand(
                    [batch_size, -1, -1]).astype(x.dtype).clone().detach()
            else:
                x = x + self.pos_embed.expand(
                    [batch_size, -1, -1]).astype(x.dtype).clone().detach()

        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias(
        ) if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            if self.lin_probe:
                if self.use_mean_pooling:
                    return self.fc_norm(t.mean(1), is_train=is_train)
                else:
                    return self.fc_norm(x[:, 0], is_train=is_train)
            else:
                return self.fc_norm(t.mean(1))

        else:
            return x[:, 0]

    def forward(self, x, is_train=True):
        x = self.forward_features(x, is_train)
        x = self.head(x)
        return x


def _enable_linear_eval(model):
    zeros_ = nn.initializer.Constant(value=0.)
    normal_ = nn.initializer.Normal(mean=0.0, std=0.01)
    linear_keyword = 'head'
    head_norm = 'fc_norm'
    requires_grad = []
    for name, param in model.named_parameters():
        if name not in [
                '%s.weight' % linear_keyword, '%s.bias' % linear_keyword
        ] and head_norm not in name:
            param.stop_gradient = True
        else:
            requires_grad.append(name)
    # init the fc layer
    normal_(getattr(model, linear_keyword).weight)
    zeros_(getattr(model, linear_keyword).bias)

    return


def _load_pretrained(pretrained,
                     pretrained_url,
                     model,
                     model_keys,
                     model_ema_configs,
                     use_abs_pos_emb,
                     use_rel_pos_bias,
                     use_ssld=False):
    if pretrained is False:
        return
    elif pretrained is True:
        local_weight_path = get_weights_path_from_url(pretrained_url).replace(
            ".pdparams", "")
        checkpoint = paddle.load(local_weight_path + ".pdparams")
    elif isinstance(pretrained, str):
        checkpoint = paddle.load(pretrained + ".pdparams")

    checkpoint_model = None
    for model_key in model_keys.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()
    all_keys = list(checkpoint_model.keys())
    # NOTE: remove all decoder keys
    all_keys = [key for key in all_keys if key.startswith('encoder.')]
    for key in all_keys:
        new_key = key.replace('encoder.', '')
        checkpoint_model[new_key] = checkpoint_model[key]
        checkpoint_model.pop(key)

    for key in list(checkpoint_model.keys()):
        if key.startswith('regressor_and_decoder.'):
            checkpoint_model.pop(key)
        if key.startswith('teacher_network.'):
            checkpoint_model.pop(key)

        # NOTE: replace norm with fc_norm
    for key in list(checkpoint_model.keys()):
        if key.startswith('norm.'):
            new_key = key.replace('norm.', 'fc_norm.')
            checkpoint_model[new_key] = checkpoint_model[key]
            checkpoint_model.pop(key)

    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[
                k].shape:
            del checkpoint_model[k]

    if model.use_rel_pos_bias and "rel_pos_bias.relative_position_bias_table" in checkpoint_model:
        num_layers = model.get_num_layers()
        rel_pos_bias = checkpoint_model[
            "rel_pos_bias.relative_position_bias_table"]
        for i in range(num_layers):
            checkpoint_model["blocks.%d.attn.relative_position_bias_table" %
                             i] = rel_pos_bias.clone()

        checkpoint_model.pop("rel_pos_bias.relative_position_bias_table")

    all_keys = list(checkpoint_model.keys())

    for key in all_keys:
        if "relative_position_index" in key:
            checkpoint_model.pop(key)

        if "relative_position_bias_table" in key and use_rel_pos_bias:
            rel_pos_bias = checkpoint_model[key]
            src_num_pos, num_attn_heads = rel_pos_bias.shape
            dst_num_pos, _ = model.state_dict()[key].shape
            dst_patch_shape = model.patch_embed.patch_shape
            if dst_patch_shape[0] != dst_patch_shape[1]:
                raise NotImplementedError()
            num_extra_tokens = dst_num_pos - (dst_patch_shape[0] * 2 - 1) * (
                dst_patch_shape[1] * 2 - 1)
            src_size = int((src_num_pos - num_extra_tokens)**0.5)
            dst_size = int((dst_num_pos - num_extra_tokens)**0.5)
            if src_size != dst_size:
                extra_tokens = rel_pos_bias[-num_extra_tokens:, :]
                rel_pos_bias = rel_pos_bias[:-num_extra_tokens, :]

                def geometric_progression(a, r, n):
                    return a * (1.0 - r**n) / (1.0 - r)

                left, right = 1.01, 1.5
                while right - left > 1e-6:
                    q = (left + right) / 2.0
                    gp = geometric_progression(1, q, src_size // 2)
                    if gp > dst_size // 2:
                        right = q
                    else:
                        left = q

                dis = []
                cur = 1
                for i in range(src_size // 2):
                    dis.append(cur)
                    cur += q**(i + 1)

                r_ids = [-_ for _ in reversed(dis)]

                x = r_ids + [0] + dis
                y = r_ids + [0] + dis

                t = dst_size // 2.0
                dx = np.arange(-t, t + 0.1, 1.0)
                dy = np.arange(-t, t + 0.1, 1.0)

                all_rel_pos_bias = []

                for i in range(num_attn_heads):
                    z = rel_pos_bias[:, i].view(src_size,
                                                src_size).float().numpy()
                    f = interpolate.interp2d(x, y, z, kind='cubic')
                    all_rel_pos_bias.append(
                        paddle.Tensor(f(dx, dy)).astype('float32').reshape(
                            [-1, 1]))

                rel_pos_bias = paddle.concat(all_rel_pos_bias, axis=-1)

                new_rel_pos_bias = paddle.concat(
                    (rel_pos_bias, extra_tokens), axis=0)
                checkpoint_model[key] = new_rel_pos_bias

    # interpolate position embedding
    if 'pos_embed' in checkpoint_model and use_abs_pos_emb:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**
                        0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size,
                                            embedding_size).permute(0, 3, 1, 2)
            pos_tokens = paddle.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode='bicubic',
                align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = paddle.concat((extra_tokens, pos_tokens), axis=1)
            checkpoint_model['pos_embed'] = new_pos_embed
    msg = model.set_state_dict(checkpoint_model)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters()
                       if not p.stop_gradient).item()

    return


def cae_base_patch16_224(pretrained=True, use_ssld=False, **kwargs):
    config = kwargs.copy()
    enable_linear_eval = config.pop('enable_linear_eval')
    model_keys = config.pop('model_key')
    model_ema_configs = config.pop('model_ema')
    use_abs_pos_emb = config.get('use_abs_pos_emb', False)
    use_rel_pos_bias = config.get('use_rel_pos_bias', True)
    if pretrained in config:
        pretrained = config.pop('pretrained')

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **config)

    if enable_linear_eval:
        _enable_linear_eval(model)

    _load_pretrained(
        pretrained,
        MODEL_URLS["cae_base_patch16_224"],
        model,
        model_keys,
        model_ema_configs,
        use_abs_pos_emb,
        use_rel_pos_bias,
        use_ssld=False)

    return model


def cae_large_patch16_224(pretrained=True, use_ssld=False, **kwargs):
    config = kwargs.copy()
    enable_linear_eval = config.pop('enable_linear_eval')
    model_keys = config.pop('model_key')
    model_ema_configs = config.pop('model_ema')
    use_abs_pos_emb = config.get('use_abs_pos_emb', False)
    use_rel_pos_bias = config.get('use_rel_pos_bias', True)
    if pretrained in config:
        pretrained = config.pop('pretrained')

    model = VisionTransformer(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        **config)

    if enable_linear_eval:
        _enable_linear_eval(model)

    _load_pretrained(
        pretrained,
        MODEL_URLS["cae_large_patch16_224"],
        model,
        model_keys,
        model_ema_configs,
        use_abs_pos_emb,
        use_rel_pos_bias,
        use_ssld=False)

    return model
