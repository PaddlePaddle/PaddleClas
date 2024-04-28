# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/micronDLA/MobileViTv3/blob/main/MobileViTv3-v1/cvnets/models/classification/mobilevit.py
# reference: https://arxiv.org/abs/2209.15159

import math
from functools import partial
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileViTV3_XXS":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_XXS_pretrained.pdparams",
    "MobileViTV3_XS":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_XS_pretrained.pdparams",
    "MobileViTV3_S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_S_pretrained.pdparams",
    "MobileViTV3_XXS_L2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_XXS_L2_pretrained.pdparams",
    "MobileViTV3_XS_L2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_XS_L2_pretrained.pdparams",
    "MobileViTV3_S_L2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_S_L2_pretrained.pdparams",
    "MobileViTV3_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_x0_5_pretrained.pdparams",
    "MobileViTV3_x0_75":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_x0_75_pretrained.pdparams",
    "MobileViTV3_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV3_x1_0_pretrained.pdparams",
}

layer_norm_2d = partial(nn.GroupNorm, num_groups=1)


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class InvertedResidual(nn.Layer):
    """
    Inverted residual block (MobileNetv2): https://arxiv.org/abs/1801.04381
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int,
                 expand_ratio: Union[int, float],
                 dilation: int=1) -> None:
        assert stride in [1, 2]
        super(InvertedResidual, self).__init__()
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_sublayer(
                name="exp_1x1",
                sublayer=nn.Sequential(
                    ('conv', nn.Conv2D(
                        in_channels, hidden_dim, 1, bias_attr=False)),
                    ('norm', nn.BatchNorm2D(hidden_dim)), ('act', nn.Silu())))

        block.add_sublayer(
            name="conv_3x3",
            sublayer=nn.Sequential(
                ('conv', nn.Conv2D(
                    hidden_dim,
                    hidden_dim,
                    3,
                    bias_attr=False,
                    stride=stride,
                    padding=dilation,
                    dilation=dilation,
                    groups=hidden_dim)), ('norm', nn.BatchNorm2D(hidden_dim)),
                ('act', nn.Silu())))

        block.add_sublayer(
            name="red_1x1",
            sublayer=nn.Sequential(
                ('conv', nn.Conv2D(
                    hidden_dim, out_channels, 1, bias_attr=False)),
                ('norm', nn.BatchNorm2D(out_channels))))

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.dilation = dilation

    def forward(self, x, *args, **kwargs):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


class MultiHeadAttention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv_proj = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias_attr=qkv_bias)

    def forward(self, x):
        # B = x.shape[0]
        N, C = x.shape[1:]
        qkv = self.qkv_proj(x).reshape((-1, N, 3, self.num_heads,
                                        C // self.num_heads)).transpose(
                                            (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((-1, N, C))
        x = self.out_proj(x)
        return x


class TransformerEncoder(nn.Layer):
    """
        This class defines the Transformer encoder (pre-norm) as described in "Attention is all you need" paper
            https://arxiv.org/abs/1706.03762
    """

    def __init__(self,
                 embed_dim: int,
                 ffn_latent_dim: int,
                 num_heads: Optional[int]=8,
                 attn_dropout: Optional[float]=0.0,
                 dropout: Optional[float]=0.1,
                 ffn_dropout: Optional[float]=0.0,
                 transformer_norm_layer: nn.Layer=nn.LayerNorm):
        super(TransformerEncoder, self).__init__()

        self.pre_norm_mha = nn.Sequential(
            transformer_norm_layer(embed_dim),
            MultiHeadAttention(
                embed_dim, num_heads, attn_drop=attn_dropout, qkv_bias=True),
            nn.Dropout(p=dropout))

        self.pre_norm_ffn = nn.Sequential(
            transformer_norm_layer(embed_dim),
            nn.Linear(embed_dim, ffn_latent_dim),
            nn.Silu(),
            nn.Dropout(p=ffn_dropout),
            nn.Linear(ffn_latent_dim, embed_dim),
            nn.Dropout(p=dropout))
        self.embed_dim = embed_dim
        self.ffn_dim = ffn_latent_dim
        self.ffn_dropout = ffn_dropout

    def forward(self, x):
        # Multi-head attention
        x = x + self.pre_norm_mha(x)

        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTV3Block(nn.Layer):
    """
        MobileViTV3 block
    """

    def __init__(self,
                 in_channels: int,
                 transformer_dim: int,
                 ffn_dim: int,
                 n_transformer_blocks: Optional[int]=2,
                 head_dim: Optional[int]=32,
                 attn_dropout: Optional[float]=0.1,
                 dropout: Optional[int]=0.1,
                 ffn_dropout: Optional[int]=0.1,
                 patch_h: Optional[int]=8,
                 patch_w: Optional[int]=8,
                 transformer_norm_layer: nn.Layer=nn.LayerNorm,
                 conv_ksize: Optional[int]=3,
                 dilation: Optional[int]=1,
                 var_ffn: Optional[bool]=False,
                 no_fusion: Optional[bool]=False):

        # For MobileViTV3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        padding = (conv_ksize - 1) // 2 * dilation
        conv_3x3_in = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels,
                in_channels,
                conv_ksize,
                bias_attr=False,
                padding=padding,
                dilation=dilation,
                groups=in_channels)), ('norm', nn.BatchNorm2D(in_channels)),
            ('act', nn.Silu()))
        conv_1x1_in = nn.Sequential(('conv', nn.Conv2D(
            in_channels, transformer_dim, 1, bias_attr=False)))

        conv_1x1_out = nn.Sequential(
            ('conv', nn.Conv2D(
                transformer_dim, in_channels, 1, bias_attr=False)),
            ('norm', nn.BatchNorm2D(in_channels)), ('act', nn.Silu()))
        conv_3x3_out = None

        # For MobileViTV3: input+global --> local+global
        if not no_fusion:
            #input_ch = tr_dim + in_ch
            conv_3x3_out = nn.Sequential(
                ('conv', nn.Conv2D(
                    transformer_dim + in_channels,
                    in_channels,
                    1,
                    bias_attr=False)), ('norm', nn.BatchNorm2D(in_channels)),
                ('act', nn.Silu()))

        super().__init__()
        self.local_rep = nn.Sequential()
        self.local_rep.add_sublayer(name="conv_3x3", sublayer=conv_3x3_in)
        self.local_rep.add_sublayer(name="conv_1x1", sublayer=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dims[block_idx],
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                transformer_norm_layer=transformer_norm_layer)
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(transformer_norm_layer(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out

        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.var_ffn = var_ffn
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, feature_map):
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            feature_map = F.interpolate(
                feature_map,
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape([
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w,
            patch_w
        ])
        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = reshaped_fm.transpose([0, 2, 1, 3])
        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            [batch_size, in_channels, num_patches, patch_area])
        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = reshaped_fm.transpose([0, 3, 2, 1])
        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(
            [batch_size * patch_area, num_patches, in_channels])

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h
        }

        return patches, info_dict

    def folding(self, patches, info_dict):
        n_dim = patches.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            patches.shape)
        # [BP, N, C] --> [B, P, N, C]
        patches = patches.reshape([
            info_dict["batch_size"], self.patch_area,
            info_dict["total_patches"], patches.shape[2]
        ])

        batch_size, pixels, num_patches, channels = patches.shape
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = patches.transpose([0, 3, 2, 1])

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape([
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h,
            self.patch_w
        ])
        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = feature_map.transpose([0, 2, 1, 3])
        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape([
            batch_size, channels, num_patch_h * self.patch_h,
            num_patch_w * self.patch_w
        ])
        if info_dict["interpolate"]:
            feature_map = F.interpolate(
                feature_map,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False)
        return feature_map

    def forward(self, x):
        res = x

        # For MobileViTV3: Normal 3x3 convolution --> Depthwise 3x3 convolution
        fm_conv = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm_conv)

        # learn global representations
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            # For MobileViTV3: input+global --> local+global
            fm = self.fusion(paddle.concat((fm_conv, fm), axis=1))

        # For MobileViTV3: Skip connection
        fm = fm + res

        return fm


class LinearSelfAttention(nn.Layer):
    def __init__(self, embed_dim, attn_dropout=0.0, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.qkv_proj = nn.Conv2D(
            embed_dim, 1 + (2 * embed_dim), 1, bias_attr=bias)
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Conv2D(embed_dim, embed_dim, 1, bias_attr=bias)

    def forward(self, x):
        # [B, C, P, N] --> [B, h + 2d, P, N]
        qkv = self.qkv_proj(x)

        # Project x into query, key and value
        # Query --> [B, 1, P, N]
        # value, key --> [B, d, P, N]
        query, key, value = paddle.split(
            qkv, [1, self.embed_dim, self.embed_dim], axis=1)

        # apply softmax along N dimension
        context_scores = F.softmax(query, axis=-1)
        # Uncomment below line to visualize context scores
        # self.visualize_context_scores(context_scores=context_scores)
        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, d, P, N] x [B, 1, P, N] -> [B, d, P, N]
        context_vector = key * context_scores
        # [B, d, P, N] --> [B, d, P, 1]
        context_vector = paddle.sum(context_vector, axis=-1, keepdim=True)

        # combine context vector with values
        # [B, d, P, N] * [B, d, P, 1] --> [B, d, P, N]
        out = F.relu(value) * context_vector
        out = self.out_proj(out)
        return out


class LinearAttnFFN(nn.Layer):
    def __init__(self,
                 embed_dim: int,
                 ffn_latent_dim: int,
                 attn_dropout: Optional[float]=0.0,
                 dropout: Optional[float]=0.1,
                 ffn_dropout: Optional[float]=0.0,
                 norm_layer: Optional[str]=layer_norm_2d) -> None:
        super().__init__()
        attn_unit = LinearSelfAttention(
            embed_dim=embed_dim, attn_dropout=attn_dropout, bias=True)

        self.pre_norm_attn = nn.Sequential(
            norm_layer(num_channels=embed_dim),
            attn_unit,
            nn.Dropout(p=dropout))

        self.pre_norm_ffn = nn.Sequential(
            norm_layer(num_channels=embed_dim),
            nn.Conv2D(embed_dim, ffn_latent_dim, 1),
            nn.Silu(),
            nn.Dropout(p=ffn_dropout),
            nn.Conv2D(ffn_latent_dim, embed_dim, 1),
            nn.Dropout(p=dropout))

    def forward(self, x):
        # self-attention
        x = x + self.pre_norm_attn(x)
        # Feed forward network
        x = x + self.pre_norm_ffn(x)
        return x


class MobileViTV3BlockV2(nn.Layer):
    """
    This class defines the `MobileViTV3 block`
    """

    def __init__(self,
                 in_channels: int,
                 attn_unit_dim: int,
                 ffn_multiplier: float=2.0,
                 n_attn_blocks: Optional[int]=2,
                 attn_dropout: Optional[float]=0.0,
                 dropout: Optional[float]=0.0,
                 ffn_dropout: Optional[float]=0.0,
                 patch_h: Optional[int]=8,
                 patch_w: Optional[int]=8,
                 conv_ksize: Optional[int]=3,
                 dilation: Optional[int]=1,
                 attn_norm_layer: Optional[str]=layer_norm_2d):
        cnn_out_dim = attn_unit_dim

        padding = (conv_ksize - 1) // 2 * dilation
        conv_3x3_in = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels,
                in_channels,
                conv_ksize,
                bias_attr=False,
                padding=padding,
                dilation=dilation,
                groups=in_channels)), ('norm', nn.BatchNorm2D(in_channels)),
            ('act', nn.Silu()))
        conv_1x1_in = nn.Sequential(('conv', nn.Conv2D(
            in_channels, cnn_out_dim, 1, bias_attr=False)))

        super().__init__()
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer)

        # MobileViTV3: input changed from just global to local+global
        self.conv_proj = nn.Sequential(
            ('conv', nn.Conv2D(
                2 * cnn_out_dim, in_channels, 1, bias_attr=False)),
            ('norm', nn.BatchNorm2D(in_channels)))

        self.patch_h = patch_h
        self.patch_w = patch_w

    def _build_attn_layer(self,
                          d_model: int,
                          ffn_mult: float,
                          n_layers: int,
                          attn_dropout: float,
                          dropout: float,
                          ffn_dropout: float,
                          attn_norm_layer: nn.Layer):

        # ensure that dims are multiple of 16
        ffn_dims = [ffn_mult * d_model // 16 * 16] * n_layers

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer) for block_idx in range(n_layers)
        ]
        global_rep.append(attn_norm_layer(num_channels=d_model))

        return nn.Sequential(*global_rep), d_model

    def unfolding(self, feature_map):
        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_sizes=[self.patch_h, self.patch_w],
            strides=[self.patch_h, self.patch_w])
        n_patches = img_h * img_w // (self.patch_h * self.patch_w)
        patches = patches.reshape(
            [batch_size, in_channels, self.patch_h * self.patch_w, n_patches])

        return patches, (img_h, img_w)

    def folding(self, patches, output_size: Tuple[int, int]):
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape([batch_size, in_dim * patch_size, n_patches])

        feature_map = F.fold(
            patches,
            output_size,
            kernel_sizes=[self.patch_h, self.patch_w],
            strides=[self.patch_h, self.patch_w])

        return feature_map

    def forward(self, x):
        fm_conv = self.local_rep(x)

        # convert feature map to patches
        patches, output_size = self.unfolding(fm_conv)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, output_size=output_size)

        # MobileViTV3: local+global instead of only global
        fm = self.conv_proj(paddle.concat((fm, fm_conv), axis=1))

        # MobileViTV3: skip connection
        fm = fm + x

        return fm


class MobileViTV3(nn.Layer):
    """
        MobileViTV3:
    """

    def __init__(self,
                 mobilevit_config: Dict,
                 dropout=0.1,
                 class_num=1000,
                 classifier_dropout=0.1,
                 output_stride=None,
                 mobilevit_v2_based=False):
        super().__init__()
        self.round_nearest = 8
        self.dilation = 1
        self.dropout = dropout
        self.mobilevit_v2_based = mobilevit_v2_based

        dilate_l4 = dilate_l5 = False
        if output_stride == 8:
            dilate_l4 = True
            dilate_l5 = True
        elif output_stride == 16:
            dilate_l5 = True

        # store model configuration in a dictionary
        in_channels = mobilevit_config["layer0"]["img_channels"]
        out_channels = mobilevit_config["layer0"]["out_channels"]
        self.conv_1 = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels,
                out_channels,
                3,
                bias_attr=False,
                stride=2,
                padding=1)), ('norm', nn.BatchNorm2D(out_channels)),
            ('act', nn.Silu()))

        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer1"])

        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer2"])

        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=mobilevit_config["layer3"])

        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer4"],
            dilate=dilate_l4)

        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=mobilevit_config["layer5"],
            dilate=dilate_l5)

        if self.mobilevit_v2_based:
            self.conv_1x1_exp = nn.Identity()
        else:
            in_channels = out_channels
            out_channels = min(mobilevit_config["last_layer_exp_factor"] *
                               in_channels, 960)
            self.conv_1x1_exp = nn.Sequential(
                ('conv', nn.Conv2D(
                    in_channels, out_channels, 1, bias_attr=False)),
                ('norm', nn.BatchNorm2D(out_channels)), ('act', nn.Silu()))

        self.classifier = nn.Sequential()
        self.classifier.add_sublayer(
            name="global_pool",
            sublayer=nn.Sequential(nn.AdaptiveAvgPool2D(1), nn.Flatten()))
        if 0.0 < classifier_dropout < 1.0:
            self.classifier.add_sublayer(
                name="dropout", sublayer=nn.Dropout(p=classifier_dropout))
        self.classifier.add_sublayer(
            name="fc", sublayer=nn.Linear(out_channels, class_num))

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            fan_out = m.weight.shape[0] * m.weight.shape[2] * m.weight.shape[3]
            if self.mobilevit_v2_based:
                bound = 1.0 / fan_in**0.5
                nn.initializer.Uniform(-bound, bound)(m.weight)
                if m.bias is not None:
                    nn.initializer.Uniform(-bound, bound)(m.bias)
            else:
                nn.initializer.KaimingNormal(fan_in=fan_out)(m.weight)
                if m.bias is not None:
                    nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.BatchNorm2D):
            nn.initializer.Constant(1)(m.weight)
            nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Linear):
            if self.mobilevit_v2_based:
                nn.initializer.XavierUniform()(m.weight)
            else:
                nn.initializer.TruncatedNormal(std=.02)(m.weight)
            if m.bias is not None:
                nn.initializer.Constant(0)(m.bias)

    def _make_layer(self, input_channel, cfg, dilate=False):
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate)
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg)

    def _make_mit_layer(self, input_channel, cfg, dilate=False):
        prev_dilation = self.dilation
        block = []
        stride = cfg.get("stride", 1)

        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
                dilation=prev_dilation)

            block.append(layer)
            input_channel = cfg.get("out_channels")

        if self.mobilevit_v2_based:
            block.append(
                MobileViTV3BlockV2(
                    in_channels=input_channel,
                    attn_unit_dim=cfg["attn_unit_dim"],
                    ffn_multiplier=cfg.get("ffn_multiplier"),
                    n_attn_blocks=cfg.get("attn_blocks", 1),
                    ffn_dropout=0.,
                    attn_dropout=0.,
                    dilation=self.dilation,
                    patch_h=cfg.get("patch_h", 2),
                    patch_w=cfg.get("patch_w", 2)))
        else:
            head_dim = cfg.get("head_dim", 32)
            transformer_dim = cfg["transformer_channels"]
            ffn_dim = cfg.get("ffn_dim")
            if head_dim is None:
                num_heads = cfg.get("num_heads", 4)
                if num_heads is None:
                    num_heads = 4
                head_dim = transformer_dim // num_heads

            assert transformer_dim % head_dim == 0, (
                "Transformer input dimension should be divisible by head dimension. "
                "Got {} and {}.".format(transformer_dim, head_dim))

            block.append(
                MobileViTV3Block(
                    in_channels=input_channel,
                    transformer_dim=transformer_dim,
                    ffn_dim=ffn_dim,
                    n_transformer_blocks=cfg.get("transformer_blocks", 1),
                    patch_h=cfg.get("patch_h", 2),
                    patch_w=cfg.get("patch_w", 2),
                    dropout=self.dropout,
                    ffn_dropout=0.,
                    attn_dropout=0.,
                    head_dim=head_dim))

        return nn.Sequential(*block), input_channel

    def _make_mobilenet_layer(self, input_channel, cfg):
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio)
            block.append(layer)
            input_channel = output_channels
        return nn.Sequential(*block), input_channel

    def extract_features(self, x):
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)

        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = self.classifier(x)
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


def MobileViTV3_S(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 4
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 64,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 128,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 256,
            "transformer_channels": 192,
            "ffn_dim": 384,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 320,
            "transformer_channels": 240,
            "ffn_dim": 480,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_S"], use_ssld=use_ssld)
    return model


def MobileViTV3_XS(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 4
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 48,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 96,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 160,
            "transformer_channels": 120,
            "ffn_dim": 240,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 160,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_XS"], use_ssld=use_ssld)
    return model


def MobileViTV3_XXS(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 2
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 16,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 24,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 64,
            "transformer_channels": 64,
            "ffn_dim": 128,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 80,
            "transformer_channels": 80,
            "ffn_dim": 160,
            "transformer_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 128,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_XXS"], use_ssld=use_ssld)
    return model


def MobileViTV3_S_L2(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 4
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 64,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 128,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 256,
            "transformer_channels": 192,
            "ffn_dim": 384,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 320,
            "transformer_channels": 240,
            "ffn_dim": 480,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_S_L2"], use_ssld=use_ssld)
    return model


def MobileViTV3_XS_L2(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 4
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 32,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 48,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 96,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 160,
            "transformer_channels": 120,
            "ffn_dim": 240,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 160,
            "transformer_channels": 144,
            "ffn_dim": 288,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_XS_L2"], use_ssld=use_ssld)
    return model


def MobileViTV3_XXS_L2(pretrained=False, use_ssld=False, **kwargs):
    mv2_exp_mult = 2
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 16,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2"
        },
        "layer2": {
            "out_channels": 24,
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 3,
            "stride": 2,
            "block_type": "mv2"
        },
        "layer3": {  # 28x28
            "out_channels": 64,
            "transformer_channels": 64,
            "ffn_dim": 128,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer4": {  # 14x14
            "out_channels": 80,
            "transformer_channels": 80,
            "ffn_dim": 160,
            "transformer_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "layer5": {  # 7x7
            "out_channels": 128,
            "transformer_channels": 96,
            "ffn_dim": 192,
            "transformer_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "head_dim": None,
            "num_heads": 4,
            "block_type": "mobilevit"
        },
        "last_layer_exp_factor": 4
    }

    model = MobileViTV3(mobilevit_config, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_XXS_L2"], use_ssld=use_ssld)
    return model


def MobileViTV3_x1_0(pretrained=False, use_ssld=False, **kwargs):
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 32,
        },
        "layer1": {
            "out_channels": 64,
            "expand_ratio": 2,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": 128,
            "expand_ratio": 2,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": 256,
            "attn_unit_dim": 128,
            "ffn_multiplier": 2,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": 384,
            "attn_unit_dim": 192,
            "ffn_multiplier": 2,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": 512,
            "attn_unit_dim": 256,
            "ffn_multiplier": 2,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    model = MobileViTV3(mobilevit_config, mobilevit_v2_based=True, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_x1_0"], use_ssld=use_ssld)
    return model


def MobileViTV3_x0_75(pretrained=False, use_ssld=False, **kwargs):
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 24,
        },
        "layer1": {
            "out_channels": 48,
            "expand_ratio": 2,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": 96,
            "expand_ratio": 2,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": 192,
            "attn_unit_dim": 96,
            "ffn_multiplier": 2,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": 288,
            "attn_unit_dim": 144,
            "ffn_multiplier": 2,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": 384,
            "attn_unit_dim": 192,
            "ffn_multiplier": 2,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    model = MobileViTV3(mobilevit_config, mobilevit_v2_based=True, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_x0_75"], use_ssld=use_ssld)
    return model


def MobileViTV3_x0_5(pretrained=False, use_ssld=False, **kwargs):
    mobilevit_config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": 16,
        },
        "layer1": {
            "out_channels": 32,
            "expand_ratio": 2,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": 64,
            "expand_ratio": 2,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": 128,
            "attn_unit_dim": 64,
            "ffn_multiplier": 2,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": 192,
            "attn_unit_dim": 96,
            "ffn_multiplier": 2,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": 256,
            "attn_unit_dim": 128,
            "ffn_multiplier": 2,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": 2,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    model = MobileViTV3(mobilevit_config, mobilevit_v2_based=True, **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV3_x0_5"], use_ssld=use_ssld)
    return model
