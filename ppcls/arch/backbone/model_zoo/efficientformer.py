#ght (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/snap-research/EfficientFormer/blob/main/models/efficientformer.py
# reference: https://arxiv.org/abs/2206.01191

import os
import copy
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal
from typing import Dict
import itertools
from .vision_transformer import trunc_normal_, to_2tuple, DropPath, zeros_

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "EfficientFormer_L1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientFormer_L1_pretrained.pdparams",
    "EfficientFormer_L3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientFormer_L3_pretrained.pdparams",
    "EfficientFormer_L7":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/EfficientFormer_L7_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())


class Attention(nn.Layer):
    def __init__(self,
                 dim=384,
                 key_dim=32,
                 num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        self.h = self.dh + nh_kd * 2
        self.qkv = nn.Linear(dim, self.h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        self.N = N
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = self.create_parameter(
            shape=[num_heads, len(attention_offsets)],
            dtype='float32',
            default_initializer=zeros_)

        self.register_buffer(
            'attention_bias_idxs',
            paddle.to_tensor(
                idxs, dtype='int64').reshape(shape=[N, N]))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(
            shape=[B, N, self.num_heads, self.h // self.num_heads]).split(
                [self.key_dim, self.key_dim, self.d], axis=3)
        q = q.transpose(perm=[0, 2, 1, 3])
        k = k.transpose(perm=[0, 2, 1, 3])
        v = v.transpose(perm=[0, 2, 1, 3])

        self.attention_bias_idxs = paddle.flatten(
            self.attention_bias_idxs, start_axis=0, stop_axis=-1)
        attn = ((q @k.transpose(
            (0, 1, 3, 2))) * self.scale + paddle.index_select(
                self.attention_biases, self.attention_bias_idxs,
                axis=1).reshape((self.num_heads, self.N, self.N)))
        attn = F.softmax(attn, axis=-1)
        x = (attn @v).transpose(perm=[0, 2, 1, 3]).reshape(
            shape=[B, N, self.dh])
        x = self.proj(x)
        return x


def stem(in_chs, out_chs):
    sequential = nn.Sequential(
        nn.Conv2D(
            in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs // 2),
        nn.ReLU(),
        nn.Conv2D(
            out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(out_chs),
        nn.ReLU())
    return sequential


class Embedding(nn.Layer):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self,
                 patch_size=16,
                 stride=16,
                 padding=0,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=nn.BatchNorm2D):
        super().__init__()
        patch_size, stride, padding = map(to_2tuple,
                                          [patch_size, stride, padding])
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Flat(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        return x


class Pooling(nn.Layer):
    """
    Implementation of pooling for PoolFormer
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2D(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        return self.pool(x) - x


class LinearMlp(nn.Layer):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

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
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(nn.Layer):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm2D(hidden_features)
        self.norm2 = nn.BatchNorm2D(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.drop(x)
        return x


class Meta3D(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(
                    value=layer_scale_init_value))
            self.layer_scale_2 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(
                    value=layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0) *
                self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0) *
                self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Layer):
    def __init__(self,
                 dim,
                 pool_size=3,
                 mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0.,
                 drop_path=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5):
        super().__init__()

        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(
                    value=layer_scale_init_value))
            self.layer_scale_2 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(
                    value=layer_scale_init_value))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim,
                index,
                layers,
                pool_size=3,
                mlp_ratio=4.,
                act_layer=nn.GELU,
                norm_layer=nn.LayerNorm,
                drop_rate=.0,
                drop_path_rate=0.,
                use_layer_scale=True,
                layer_scale_init_value=1e-5,
                vit_num=1):
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat())
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
            block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(
                Meta3D(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value))
        else:
            blocks.append(
                Meta4D(
                    dim,
                    pool_size=pool_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat())

    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(nn.Layer):
    def __init__(self,
                 layers,
                 embed_dims=None,
                 mlp_ratios=4,
                 downsamples=None,
                 pool_size=3,
                 norm_layer=nn.LayerNorm,
                 act_layer=nn.GELU,
                 class_num=1000,
                 down_patch_size=3,
                 down_stride=2,
                 down_pad=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 use_layer_scale=True,
                 layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 vit_num=0,
                 distillation=True,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.class_num = class_num
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        for i in range(len(layers)):
            stage = meta_blocks(
                embed_dims[i],
                i,
                layers,
                pool_size=pool_size,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                norm_layer=norm_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1]))

        self.network = nn.LayerList(network)

        if self.fork_feat:
            # add a norm layer for each output
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    layer = nn.Identity()
                else:
                    layer = norm_layer(embed_dims[i_emb])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)
        else:
            # Classifier head
            self.norm = norm_layer(embed_dims[-1])
            self.head = nn.Linear(
                embed_dims[-1], class_num) if class_num > 0 \
                else nn.Identity()
            self.dist = distillation
            if self.dist:
                self.dist_head = nn.Linear(
                    embed_dims[-1], class_num) if class_num > 0 \
                    else nn.Identity()
        self.apply(self.cls_init_weights)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # output features of four stages for dense prediction
            return x
        x = self.norm(x)
        if self.dist:
            cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
            if not self.training:
                cls_out = (cls_out[0] + cls_out[1]) / 2
        else:
            cls_out = self.head(x.mean(-2))
        return cls_out


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


def EfficientFormer_L1(pretrained=False, use_ssld=False, **kwargs):
    model = EfficientFormer(
        layers=[3, 2, 6, 4],
        embed_dims=[48, 96, 224, 448],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["EfficientFormer_L1"], use_ssld=use_ssld)
    return model


def EfficientFormer_L3(pretrained=False, use_ssld=False, **kwargs):
    model = EfficientFormer(
        layers=[4, 4, 12, 6],
        embed_dims=[64, 128, 320, 512],
        downsamples=[True, True, True, True],
        vit_num=4,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["EfficientFormer_L3"], use_ssld=use_ssld)
    return model


def EfficientFormer_L7(pretrained=False, use_ssld=False, **kwargs):
    model = EfficientFormer(
        layers=[6, 6, 18, 8],
        embed_dims=[96, 192, 384, 768],
        downsamples=[True, True, True, True],
        vit_num=8,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["EfficientFormer_L7"], use_ssld=use_ssld)
    return model
