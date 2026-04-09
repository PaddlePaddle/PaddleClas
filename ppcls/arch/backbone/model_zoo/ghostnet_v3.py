# copyright (c) 2026 PaddlePaddle Authors. All Rights Reserve.
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

# reference:
# https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/ghostnet.py

from __future__ import absolute_import, division, print_function

import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base.theseus_layer import Identity
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "GhostNetV3_x0_5": "",
    "GhostNetV3_x1_0": "",
    "GhostNetV3_x1_3": "",
    "GhostNetV3_x1_6": "",
}

__all__ = list(MODEL_URLS.keys())


def make_divisible(v, divisor=4, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


def _load_pretrained(pretrained, model, model_url):
    if pretrained is False:
        return
    if pretrained is True:
        load_dygraph_pretrain(model, model_url)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


class SqueezeExcite(nn.Layer):
    def __init__(self, in_chs, rd_ratio=0.25):
        super().__init__()
        rd_channels = make_divisible(in_chs * rd_ratio, divisor=4)
        self.conv_reduce = nn.Conv2D(in_chs, rd_channels, kernel_size=1, bias_attr=True)
        self.act1 = nn.ReLU()
        self.conv_expand = nn.Conv2D(rd_channels, in_chs, kernel_size=1, bias_attr=True)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean(axis=(2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Layer):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size,
        stride=1,
        pad_type=0,
        group_size=0,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        groups = 1 if not group_size else in_chs // group_size
        self.conv = nn.Conv2D(
            in_chs,
            out_chs,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad_type,
            groups=groups,
            bias_attr=False,
        )
        self.bn1 = nn.BatchNorm2D(out_chs)
        self.act = act_layer() if act_layer is not None else Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act(x)
        return x


class GhostModuleV3(nn.Layer):
    def __init__(
        self,
        in_chs,
        out_chs,
        kernel_size=1,
        ratio=2,
        dw_size=3,
        stride=1,
        act_layer=nn.ReLU,
        mode="original",
    ):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        self.out_chs = out_chs
        init_chs = int(math.ceil(out_chs / ratio))
        new_chs = init_chs * (ratio - 1)
        self.mode = mode

        self.primary_rpr_conv = nn.LayerList(
            [
                ConvBnAct(
                    in_chs,
                    init_chs,
                    kernel_size,
                    stride,
                    pad_type=kernel_size // 2,
                    act_layer=None,
                )
                for _ in range(3)
            ]
        )
        self.primary_activation = act_layer()

        self.cheap_rpr_skip = nn.BatchNorm2D(init_chs)
        self.cheap_rpr_conv = nn.LayerList(
            [
                ConvBnAct(
                    init_chs,
                    new_chs,
                    dw_size,
                    1,
                    pad_type=dw_size // 2,
                    group_size=1,
                    act_layer=None,
                )
                for _ in range(3)
            ]
        )
        self.cheap_rpr_scale = ConvBnAct(
            init_chs,
            new_chs,
            1,
            1,
            pad_type=0,
            group_size=1,
            act_layer=None,
        )
        self.cheap_activation = act_layer()

        if self.mode == "shortcut":
            self.short_conv = nn.Sequential(
                nn.Conv2D(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=kernel_size // 2,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_chs),
                nn.Conv2D(
                    out_chs,
                    out_chs,
                    kernel_size=(1, 5),
                    stride=1,
                    padding=(0, 2),
                    groups=out_chs,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_chs),
                nn.Conv2D(
                    out_chs,
                    out_chs,
                    kernel_size=(5, 1),
                    stride=1,
                    padding=(2, 0),
                    groups=out_chs,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_chs),
            )
        else:
            self.short_conv = Identity()

    def forward(self, x):
        x1 = 0
        for branch in self.primary_rpr_conv:
            x1 = x1 + branch(x)
        x1 = self.primary_activation(x1)

        x2 = self.cheap_rpr_scale(x1) + self.cheap_rpr_skip(x1)
        for branch in self.cheap_rpr_conv:
            x2 = x2 + branch(x1)
        x2 = self.cheap_activation(x2)

        out = paddle.concat([x1, x2], axis=1)
        if self.mode != "shortcut":
            return out

        res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
        gate = F.interpolate(self.gate_fn(res), size=out.shape[-2:], mode="nearest")
        return out[:, : self.out_chs, :, :] * gate


class GhostBottleneckV3(nn.Layer):
    def __init__(
        self,
        in_chs,
        mid_chs,
        out_chs,
        dw_kernel_size=3,
        stride=1,
        act_layer=nn.ReLU,
        se_ratio=0.0,
        mode="original",
    ):
        super().__init__()
        self.stride = stride
        self.ghost1 = GhostModuleV3(
            in_chs,
            mid_chs,
            act_layer=act_layer,
            mode=mode,
        )

        if self.stride > 1:
            self.dw_rpr_conv = nn.LayerList(
                [
                    ConvBnAct(
                        mid_chs,
                        mid_chs,
                        dw_kernel_size,
                        stride,
                        pad_type=(dw_kernel_size - 1) // 2,
                        group_size=1,
                        act_layer=None,
                    )
                    for _ in range(3)
                ]
            )
            self.dw_rpr_scale = ConvBnAct(
                mid_chs,
                mid_chs,
                1,
                2,
                pad_type=0,
                group_size=1,
                act_layer=None,
            )
        else:
            self.dw_rpr_conv = nn.LayerList()
            self.dw_rpr_scale = Identity()

        self.se = (
            SqueezeExcite(mid_chs, rd_ratio=se_ratio) if se_ratio > 0 else Identity()
        )
        self.ghost2 = GhostModuleV3(
            mid_chs,
            out_chs,
            act_layer=Identity,
            mode="original",
        )

        if in_chs == out_chs and self.stride == 1:
            self.shortcut = Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2D(
                    in_chs,
                    in_chs,
                    dw_kernel_size,
                    stride=stride,
                    padding=(dw_kernel_size - 1) // 2,
                    groups=in_chs,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(in_chs),
                nn.Conv2D(
                    in_chs,
                    out_chs,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias_attr=False,
                ),
                nn.BatchNorm2D(out_chs),
            )

    def forward(self, x):
        shortcut = x
        x = self.ghost1(x)

        if self.stride > 1:
            x1 = self.dw_rpr_scale(x)
            for branch in self.dw_rpr_conv:
                x1 = x1 + branch(x)
            x = x1

        x = self.se(x)
        x = self.ghost2(x)
        x = x + self.shortcut(shortcut)
        return x


class GhostNetV3(nn.Layer):
    def __init__(self, width=1.0, class_num=1000, in_chans=3, drop_rate=0.2):
        super().__init__()
        self.cfgs = [
            [[3, 16, 16, 0, 1]],
            [[3, 48, 24, 0, 2]],
            [[3, 72, 24, 0, 1]],
            [[5, 72, 40, 0.25, 2]],
            [[5, 120, 40, 0.25, 1]],
            [[3, 240, 80, 0, 2]],
            [
                [3, 200, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 184, 80, 0, 1],
                [3, 480, 112, 0.25, 1],
                [3, 672, 112, 0.25, 1],
            ],
            [[5, 672, 160, 0.25, 2]],
            [
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
                [5, 960, 160, 0, 1],
                [5, 960, 160, 0.25, 1],
            ],
        ]
        self.drop_rate = drop_rate
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv2D(
            in_chans, stem_chs, 3, stride=2, padding=1, bias_attr=False
        )
        self.bn1 = nn.BatchNorm2D(stem_chs)
        self.act1 = nn.ReLU()

        prev_chs = stem_chs
        stages = []
        layer_idx = 0
        exp_size = 0
        for cfg in self.cfgs:
            layers = []
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                mode = "shortcut" if layer_idx > 1 else "original"
                layers.append(
                    GhostBottleneckV3(
                        prev_chs,
                        mid_chs,
                        out_chs,
                        dw_kernel_size=k,
                        stride=s,
                        act_layer=nn.ReLU,
                        se_ratio=se_ratio,
                        mode=mode,
                    )
                )
                prev_chs = out_chs
                layer_idx += 1
            stages.append(nn.Sequential(*layers))

        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(ConvBnAct(prev_chs, out_chs, 1)))
        self.num_features = out_chs
        self.blocks = nn.Sequential(*stages)

        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.conv_head = nn.Conv2D(
            out_chs, 1280, 1, stride=1, padding=0, bias_attr=True
        )
        self.act2 = nn.ReLU()
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.classifier = nn.Linear(1280, class_num) if class_num > 0 else Identity()

    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.0:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def GhostNetV3_x0_5(pretrained=False, **kwargs):
    model = GhostNetV3(width=0.5, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["GhostNetV3_x0_5"])
    return model


def GhostNetV3_x1_0(pretrained=False, **kwargs):
    model = GhostNetV3(width=1.0, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["GhostNetV3_x1_0"])
    return model


def GhostNetV3_x1_3(pretrained=False, **kwargs):
    model = GhostNetV3(width=1.3, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["GhostNetV3_x1_3"])
    return model


def GhostNetV3_x1_6(pretrained=False, **kwargs):
    model = GhostNetV3(width=1.6, **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["GhostNetV3_x1_6"])
    return model
