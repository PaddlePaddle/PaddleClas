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

# Code was based on https://github.com/apple/ml-cvnets/blob/7be93d3debd45c240a058e3f34a9e88d33c07a7d/cvnets/models/classification/mobilevit_v2.py
# reference: https://arxiv.org/abs/2206.02680

from functools import partial
from typing import Dict, Optional, Tuple, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileViTV2_x0_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV2_x0_5_pretrained.pdparams",
    "MobileViTV2_x1_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV2_x1_0_pretrained.pdparams",
    "MobileViTV2_x1_5":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV2_x1_5_pretrained.pdparams",
    "MobileViTV2_x2_0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViTV2_x2_0_pretrained.pdparams",
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
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 skip_connection=True):
        super().__init__()
        assert stride in [1, 2]
        self.stride = stride

        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels and skip_connection

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

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)


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
                 embed_dim,
                 ffn_latent_dim,
                 attn_dropout=0.0,
                 dropout=0.1,
                 ffn_dropout=0.0,
                 norm_layer=layer_norm_2d) -> None:
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


class MobileViTV2Block(nn.Layer):
    """
    This class defines the `MobileViTV2 block`
    """

    def __init__(self,
                 in_channels,
                 attn_unit_dim,
                 ffn_multiplier=2.0,
                 n_attn_blocks=2,
                 attn_dropout=0.0,
                 dropout=0.0,
                 ffn_dropout=0.0,
                 patch_h=8,
                 patch_w=8,
                 conv_ksize=3,
                 dilation=1,
                 attn_norm_layer=layer_norm_2d):
        super().__init__()
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

        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer)

        self.conv_proj = nn.Sequential(
            ('conv', nn.Conv2D(
                cnn_out_dim, in_channels, 1, bias_attr=False)),
            ('norm', nn.BatchNorm2D(in_channels)))

        self.patch_h = patch_h
        self.patch_w = patch_w

    def _build_attn_layer(self, d_model, ffn_mult, n_layers, attn_dropout,
                          dropout, ffn_dropout, attn_norm_layer):
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

    def folding(self, patches, output_size):
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
        fm = self.local_rep(x)

        # convert feature map to patches
        patches, output_size = self.unfolding(fm)

        # learn global representations on all patches
        patches = self.global_rep(patches)

        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm


class MobileViTV2(nn.Layer):
    """
        MobileViTV2
    """

    def __init__(self, mobilevit_config, class_num=1000, output_stride=None):
        super().__init__()
        self.round_nearest = 8
        self.dilation = 1

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

        self.conv_1x1_exp = nn.Identity()
        self.classifier = nn.Sequential()
        self.classifier.add_sublayer(
            name="global_pool",
            sublayer=nn.Sequential(nn.AdaptiveAvgPool2D(1), nn.Flatten()))
        self.classifier.add_sublayer(
            name="fc", sublayer=nn.Linear(out_channels, class_num))

        # weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            fan_in = m.weight.shape[1] * m.weight.shape[2] * m.weight.shape[3]
            bound = 1.0 / fan_in**0.5
            nn.initializer.Uniform(-bound, bound)(m.weight)
            if m.bias is not None:
                nn.initializer.Uniform(-bound, bound)(m.bias)
        elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
            nn.initializer.Constant(1)(m.weight)
            nn.initializer.Constant(0)(m.bias)
        elif isinstance(m, nn.Linear):
            nn.initializer.XavierUniform()(m.weight)
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

        block.append(
            MobileViTV2Block(
                in_channels=input_channel,
                attn_unit_dim=cfg["attn_unit_dim"],
                ffn_multiplier=cfg.get("ffn_multiplier"),
                n_attn_blocks=cfg.get("attn_blocks", 1),
                ffn_dropout=0.,
                attn_dropout=0.,
                dilation=self.dilation,
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2)))

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


def get_configuration(width_multiplier):
    ffn_multiplier = 2
    mv2_exp_mult = 2  # max(1.0, min(2.0, 2.0 * width_multiplier))

    layer_0_dim = max(16, min(64, 32 * width_multiplier))
    layer_0_dim = int(make_divisible(layer_0_dim, divisor=8, min_value=16))
    config = {
        "layer0": {
            "img_channels": 3,
            "out_channels": layer_0_dim,
        },
        "layer1": {
            "out_channels": int(make_divisible(64 * width_multiplier, divisor=16)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 1,
            "stride": 1,
            "block_type": "mv2",
        },
        "layer2": {
            "out_channels": int(make_divisible(128 * width_multiplier, divisor=8)),
            "expand_ratio": mv2_exp_mult,
            "num_blocks": 2,
            "stride": 2,
            "block_type": "mv2",
        },
        "layer3": {  # 28x28
            "out_channels": int(make_divisible(256 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(128 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 2,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer4": {  # 14x14
            "out_channels": int(make_divisible(384 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(192 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 4,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "layer5": {  # 7x7
            "out_channels": int(make_divisible(512 * width_multiplier, divisor=8)),
            "attn_unit_dim": int(make_divisible(256 * width_multiplier, divisor=8)),
            "ffn_multiplier": ffn_multiplier,
            "attn_blocks": 3,
            "patch_h": 2,
            "patch_w": 2,
            "stride": 2,
            "mv_expand_ratio": mv2_exp_mult,
            "block_type": "mobilevit",
        },
        "last_layer_exp_factor": 4,
    }

    return config


def MobileViTV2_x2_0(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 2.0
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x2_0"], use_ssld=use_ssld)
    return model


def MobileViTV2_x1_75(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 1.75
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x1_75"], use_ssld=use_ssld)
    return model


def MobileViTV2_x1_5(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 1.5
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x1_5"], use_ssld=use_ssld)
    return model


def MobileViTV2_x1_25(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 1.25
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x1_25"], use_ssld=use_ssld)
    return model


def MobileViTV2_x1_0(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 1.0
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x1_0"], use_ssld=use_ssld)
    return model


def MobileViTV2_x0_75(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 0.75
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x0_75"], use_ssld=use_ssld)
    return model


def MobileViTV2_x0_5(pretrained=False, use_ssld=False, **kwargs):
    width_multiplier = 0.5
    model = MobileViTV2(get_configuration(width_multiplier), **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViTV2_x0_5"], use_ssld=use_ssld)
    return model
