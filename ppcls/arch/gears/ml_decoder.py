# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import XavierNormal, Constant, Normal

xavier_normal_ = XavierNormal()
normal_ = Normal
zero_ = Constant(value=0.0)


class MLDecoder(nn.Layer):
    """
    ML-Decoder is an attention-based classification head,
    which introduced by Tal Ridnik et al. in https://arxiv.org/pdf/2111.12933.pdf.
    """

    def __init__(self,
                 class_num=80,
                 in_channels=2048,
                 query_num=80,
                 embed_dim=768,
                 depth=1,
                 num_heads=8,
                 mlp_hidden_dim=2048,
                 dropout=0.1,
                 activation="relu",
                 freeze_query_embed=True,
                 remove_self_attn=True):
        super().__init__()
        self.class_num = class_num
        self.in_channels = in_channels

        # 1 <= query_num <= class_num
        query_num = min(max(query_num, 1), class_num)

        self.input_proj = nn.Conv2D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1)

        self.query_pos_embed = nn.Embedding(
            num_embeddings=query_num,
            embedding_dim=embed_dim)
        if freeze_query_embed:
            self.query_pos_embed.weight.stop_gradient = True

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_hidden_dim,
            dropout=dropout,
            activation=activation,
            attn_dropout=dropout,
            act_dropout=dropout)
        if remove_self_attn:
            del decoder_layer.self_attn
            decoder_layer.self_attn = self.self_attn_identity
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=depth)

        group_factor = math.ceil(class_num / query_num)
        self.group_conv = nn.Conv2D(
            in_channels=query_num * embed_dim,
            out_channels=query_num * group_factor,
            kernel_size=1,
            stride=1,
            groups=query_num)

        self._init_weights()

    def _init_weights(self):
        normal_(self.query_pos_embed.weight)
        xavier_normal_(self.group_conv.weight)
        zero_(self.group_conv.bias)

    @staticmethod
    def self_attn_identity(*args):
        return args[0]

    def group_fc_pool(self, x):
        x = x.flatten(1)[..., None, None]
        x = self.group_conv(x)
        x = x.flatten(1)[:, :self.class_num]
        return x

    def forward(self, x):
        if x.ndim == 2:
            assert x.shape[1] % self.in_channels == 0, "Wrong `in_channels` value!!!"
            x = x.reshape([x.shape[0], self.in_channels, -1, 1])
        elif x.ndim == 3:
            assert x.shape[1] == self.in_channels, "Wrong input shape!!!"
            x = x.unsqueeze(-1)
        else:
            assert x.ndim == 4 and x.shape[1] == self.in_channels, "Wrong input shape!!!"

        feat_proj = F.relu(self.input_proj(x))
        feat_flatten = feat_proj.flatten(2).transpose([0, 2, 1])

        query_pos_embed = self.query_pos_embed.weight[None].tile([x.shape[0], 1, 1])
        out_embed = self.decoder(query_pos_embed, feat_flatten)

        logit = self.group_fc_pool(out_embed)
        return logit
