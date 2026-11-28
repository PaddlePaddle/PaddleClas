# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/shvit.py
# paper: SHViT, Single-Head Vision Transformer with Memory Efficient Macro Design, https://arxiv.org/abs/2401.16456

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain
from ..base.theseus_layer import TheseusLayer

MODEL_URLS = {
    "SHViT_S1":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SHViT_S1.pdparams",
    "SHViT_S2":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SHViT_S2.pdparams",
    "SHViT_S3":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SHViT_S3.pdparams",
    "SHViT_S4":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SHViT_S4.pdparams",
}

__all__ = list(MODEL_URLS.keys())


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    return max(min_value, int(v + divisor / 2) // divisor * divisor)


class Residual(nn.Layer):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class Conv2dNorm(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=1, bn_weight_init=1):
        super().__init__()
        self.add_sublayer(
            "c",
            nn.Conv2D(
                in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias_attr=False
            ),
        )
        self.add_sublayer(
            "bn",
            nn.BatchNorm2D(
                out_channels,
                weight_attr=nn.initializer.Constant(bn_weight_init),
                bias_attr=nn.initializer.Constant(0.0),
            ),
        )


class NormLinear(nn.Layer):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.bn = nn.BatchNorm1D(in_features)
        self.l = nn.Linear(in_features, out_features)
        nn.initializer.TruncatedNormal(std=0.02)(self.l.weight)
        nn.initializer.Constant(0.0)(self.l.bias)

    def forward(self, x):
        return self.l(self.bn(x))


class SqueezeExcite(nn.Layer):
    def __init__(self, channels, rd_ratio=0.25):
        super().__init__()
        rd_channels = make_divisible(channels * rd_ratio, 8)
        self.fc1 = nn.Conv2D(channels, rd_channels, kernel_size=1)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2D(rd_channels, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean(axis=[2, 3], keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class PatchMerging(nn.Layer):
    def __init__(self, dim, out_dim):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2dNorm(dim, hid_dim)
        self.act1 = nn.ReLU()
        self.conv2 = Conv2dNorm(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim)
        self.act2 = nn.ReLU()
        self.se = SqueezeExcite(hid_dim, 0.25)
        self.conv3 = Conv2dNorm(hid_dim, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.conv3(x)
        return x


class FFN(nn.Layer):
    def __init__(self, dim, embed_dim):
        super().__init__()
        self.pw1 = Conv2dNorm(dim, embed_dim)
        self.act = nn.ReLU()
        self.pw2 = Conv2dNorm(embed_dim, dim, bn_weight_init=0)

    def forward(self, x):
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x


class SHSA(nn.Layer):
    def __init__(self, dim, qk_dim, pdim):
        super().__init__()
        self.scale = qk_dim ** -0.5
        self.qk_dim = qk_dim
        self.dim = dim
        self.pdim = pdim

        self.pre_norm = nn.GroupNorm(num_groups=1, num_channels=pdim)
        self.qkv = Conv2dNorm(pdim, qk_dim * 2 + pdim)
        self.proj = nn.Sequential(nn.ReLU(), Conv2dNorm(dim, dim, bn_weight_init=0))

    def forward(self, x):
        B, _, H, W = x.shape
        x1, x2 = paddle.split(x, [self.pdim, self.dim - self.pdim], axis=1)
        x1 = self.pre_norm(x1)
        qkv = self.qkv(x1)
        q, k, v = paddle.split(qkv, [self.qk_dim, self.qk_dim, self.pdim], axis=1)
        q = q.flatten(start_axis=2)
        k = k.flatten(start_axis=2)
        v = v.flatten(start_axis=2)

        attn = (q.transpose([0, 2, 1]) @ k) * self.scale
        attn = F.softmax(attn, axis=-1)
        x1 = (v @ attn.transpose([0, 2, 1])).reshape([B, self.pdim, H, W])
        x = self.proj(paddle.concat([x1, x2], axis=1))
        return x


class BasicBlock(nn.Layer):
    def __init__(self, dim, qk_dim, pdim, block_type):
        super().__init__()
        self.conv = Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim, bn_weight_init=0))
        if block_type == "s":
            self.mixer = Residual(SHSA(dim, qk_dim, pdim))
        else:
            self.mixer = nn.Identity()
        self.ffn = Residual(FFN(dim, int(dim * 2)))

    def forward(self, x):
        x = self.conv(x)
        x = self.mixer(x)
        x = self.ffn(x)
        return x


class StageBlock(nn.Layer):
    def __init__(self, prev_dim, dim, qk_dim, pdim, block_type, depth):
        super().__init__()
        if prev_dim != dim:
            self.downsample = nn.Sequential(
                Residual(Conv2dNorm(prev_dim, prev_dim, 3, 1, 1, groups=prev_dim)),
                Residual(FFN(prev_dim, int(prev_dim * 2))),
                PatchMerging(prev_dim, dim),
                Residual(Conv2dNorm(dim, dim, 3, 1, 1, groups=dim)),
                Residual(FFN(dim, int(dim * 2))),
            )
        else:
            self.downsample = nn.Identity()

        self.blocks = nn.Sequential(
            *[BasicBlock(dim, qk_dim, pdim, block_type) for _ in range(depth)]
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class SHViT(TheseusLayer):
    def __init__(
            self,
            in_chans=3,
            class_num=1000,
            embed_dim=(128, 256, 384),
            partial_dim=(32, 64, 96),
            qk_dim=(16, 16, 16),
            depth=(1, 2, 3),
            types=("s", "s", "s"),
            drop_rate=0.,
    ):
        super().__init__()
        self.num_classes = class_num
        self.drop_rate = drop_rate

        stem_chs = embed_dim[0]
        self.patch_embed = nn.Sequential(
            Conv2dNorm(in_chans, stem_chs // 8, 3, 2, 1),
            nn.ReLU(),
            Conv2dNorm(stem_chs // 8, stem_chs // 4, 3, 2, 1),
            nn.ReLU(),
            Conv2dNorm(stem_chs // 4, stem_chs // 2, 3, 2, 1),
            nn.ReLU(),
            Conv2dNorm(stem_chs // 2, stem_chs, 3, 2, 1),
        )

        stages = []
        prev_chs = stem_chs
        for i in range(len(embed_dim)):
            stages.append(
                StageBlock(
                    prev_dim=prev_chs,
                    dim=embed_dim[i],
                    qk_dim=qk_dim[i],
                    pdim=partial_dim[i],
                    block_type=types[i],
                    depth=depth[i],
                )
            )
            prev_chs = embed_dim[i]
        self.stages = nn.Sequential(*stages)

        self.num_features = self.head_hidden_size = embed_dim[-1]
        self.head = NormLinear(self.head_hidden_size, class_num) if class_num > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = F.adaptive_avg_pool2d(x, 1)
        x = paddle.flatten(x, 1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        return
    if pretrained is True:
        if not model_url:
            raise ValueError(
                "No pretrained weights are available for this SHViT variant. "
                "Please pass a local weights path to `pretrained`."
            )
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def SHViT_S1(pretrained=False, use_ssld=False, **kwargs):
    model = SHViT(
        embed_dim=(128, 224, 320),
        depth=(2, 4, 5),
        partial_dim=(32, 48, 68),
        types=("i", "s", "s"),
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["SHViT_S1"], use_ssld)
    return model


def SHViT_S2(pretrained=False, use_ssld=False, **kwargs):
    model = SHViT(
        embed_dim=(128, 308, 448),
        depth=(2, 4, 5),
        partial_dim=(32, 66, 96),
        types=("i", "s", "s"),
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["SHViT_S2"], use_ssld)
    return model


def SHViT_S3(pretrained=False, use_ssld=False, **kwargs):
    model = SHViT(
        embed_dim=(192, 352, 448),
        depth=(3, 5, 5),
        partial_dim=(48, 75, 96),
        types=("i", "s", "s"),
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["SHViT_S3"], use_ssld)
    return model


def SHViT_S4(pretrained=False, use_ssld=False, **kwargs):
    model = SHViT(
        embed_dim=(224, 336, 448),
        depth=(4, 7, 6),
        partial_dim=(48, 72, 96),
        types=("i", "s", "s"),
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["SHViT_S4"], use_ssld)
    return model
