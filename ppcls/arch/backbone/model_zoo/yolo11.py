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

# reference: https://github.com/ultralytics/ultralytics

import math

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "YOLO11_cls_n": (
        "https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/yolo11/media/"
        "YOLO11_cls_n_pretrained.pdparams"
    ),
    "YOLO11_cls_s": (
        "https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/yolo11/media/"
        "YOLO11_cls_s_pretrained.pdparams"
    ),
    "YOLO11_cls_m": (
        "https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/yolo11/media/"
        "YOLO11_cls_m_pretrained.pdparams"
    ),
    "YOLO11_cls_l": (
        "https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/yolo11/media/"
        "YOLO11_cls_l_pretrained.pdparams"
    ),
    "YOLO11_cls_x": (
        "https://git.aistudio.baidu.com/api/v1/repos/PaddleClas/yolo11/media/"
        "YOLO11_cls_x_pretrained.pdparams"
    ),
}

__all__ = [
    "YOLO11_cls_n",
    "YOLO11_cls_s",
    "YOLO11_cls_m",
    "YOLO11_cls_l",
    "YOLO11_cls_x",
]


YOLO11_CLS_CFG = {
    "n": {
        "depth": 0.50,
        "width": 0.25,
        "max_channels": 1024,
    },
    "s": {
        "depth": 0.50,
        "width": 0.50,
        "max_channels": 1024,
    },
    "m": {
        "depth": 0.50,
        "width": 1.00,
        "max_channels": 512,
    },
    "l": {
        "depth": 1.00,
        "width": 1.00,
        "max_channels": 512,
    },
    "x": {
        "depth": 1.00,
        "width": 1.50,
        "max_channels": 512,
    },
}


def _load_pretrained(pretrained, model, model_url=None, use_ssld=False):
    if pretrained is False:
        return
    if pretrained is True:
        if not model_url:
            raise ValueError(
                "No official PaddleClas pretrained weights are available for "
                f"{model.__class__.__name__}. Please pass a local pretrained path."
            )
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
        return
    if isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
        return
    raise RuntimeError(
        "pretrained type is not available. Please use `string` or `boolean` type."
    )


def make_divisible(x, divisor=8):
    return int(math.ceil(float(x) / divisor) * divisor)


def scale_channels(channels, width, max_channels, divisor=8):
    return make_divisible(min(channels, max_channels) * width, divisor)


def scale_depth(repeats, depth):
    return max(round(repeats * depth), 1) if repeats > 1 else repeats


class ConvBNAct(nn.Layer):
    default_act = nn.Silu()

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        groups=1,
        dilation=1,
        act=True,
    ):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias_attr=False,
        )
        self.bn = nn.BatchNorm2D(out_channels, epsilon=1e-5, momentum=0.9)
        if act is True:
            self.act = self.default_act
        elif isinstance(act, nn.Layer):
            self.act = act
        else:
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Layer):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, kernel_size=k[0], stride=1)
        self.cv2 = ConvBNAct(c_, c2, kernel_size=k[1], stride=1, groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.add:
            y = x + y
        return y


class C3k(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 1, 1)
        self.cv2 = ConvBNAct(c1, c_, 1, 1)
        self.cv3 = ConvBNAct(2 * c_, c2, 1, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(paddle.concat([self.m(self.cv1(x)), self.cv2(x)], axis=1))


class C2f(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1, 1)
        self.m = nn.LayerList(
            [Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n)]
        )

    def forward(self, x):
        y = list(paddle.split(self.cv1(x), num_or_sections=[self.c, self.c], axis=1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(paddle.concat(y, axis=1))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.LayerList(
            [
                (
                    C3k(self.c, self.c, n=2, shortcut=shortcut, g=g)
                    if c3k
                    else Bottleneck(self.c, self.c, shortcut=shortcut, g=g)
                )
                for _ in range(n)
            ]
        )


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct(dim, h, 1, 1, act=False)
        self.proj = ConvBNAct(dim, dim, 1, 1, act=False)
        self.pe = ConvBNAct(dim, dim, 3, 1, groups=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(
            [b, self.num_heads, self.key_dim * 2 + self.head_dim, n]
        )
        q, k, v = paddle.split(
            qkv, num_or_sections=[self.key_dim, self.key_dim, self.head_dim], axis=2
        )
        attn = paddle.matmul(q.transpose([0, 1, 3, 2]), k) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        y = paddle.matmul(v, attn.transpose([0, 1, 3, 2])).reshape([b, c, h, w])
        y = y + self.pe(v.reshape([b, c, h, w]))
        return self.proj(y)


class PSABlock(nn.Layer):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            ConvBNAct(c, c * 2, 1, 1), ConvBNAct(c * 2, c, 1, 1, act=False)
        )
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Layer):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        if c1 != c2:
            raise ValueError(f"C2PSA expects c1 == c2, but got {c1} and {c2}")
        self.c = int(c1 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvBNAct(2 * self.c, c1, 1, 1)
        num_heads = max(self.c // 64, 1)
        self.m = nn.Sequential(
            *[PSABlock(self.c, attn_ratio=0.5, num_heads=num_heads) for _ in range(n)]
        )

    def forward(self, x):
        a, b = paddle.split(self.cv1(x), num_or_sections=[self.c, self.c], axis=1)
        b = self.m(b)
        return self.cv2(paddle.concat([a, b], axis=1))


class ClassifyHead(nn.Layer):
    def __init__(self, in_channels, class_num=1000):
        super().__init__()
        self.conv = ConvBNAct(in_channels, 1280, 1, 1)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(p=0.0)
        self.fc = nn.Linear(1280, class_num)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.dropout(x)
        return self.fc(x)


class YOLO11CLS(nn.Layer):
    def __init__(self, scale="n", class_num=1000, **kwargs):
        super().__init__()
        if scale not in YOLO11_CLS_CFG:
            raise ValueError(f"Unsupported YOLO11 cls scale: {scale}")
        self.scale = scale
        self.class_num = class_num
        cfg = YOLO11_CLS_CFG[scale]
        depth = cfg["depth"]
        width = cfg["width"]
        max_channels = cfg["max_channels"]

        c1 = scale_channels(64, width, max_channels)
        c2 = scale_channels(128, width, max_channels)
        c3 = scale_channels(256, width, max_channels)
        c4 = scale_channels(512, width, max_channels)
        c5 = scale_channels(1024, width, max_channels)

        n2 = scale_depth(2, depth)
        n4 = scale_depth(2, depth)
        n6 = scale_depth(2, depth)
        n8 = scale_depth(2, depth)
        n9 = scale_depth(2, depth)
        use_c3k_early = scale in {"m", "l", "x"}

        self.model = nn.LayerList(
            [
                ConvBNAct(3, c1, 3, 2),
                ConvBNAct(c1, c2, 3, 2),
                C3k2(c2, c3, n=n2, c3k=use_c3k_early, e=0.25),
                ConvBNAct(c3, c3, 3, 2),
                C3k2(c3, c4, n=n4, c3k=use_c3k_early, e=0.25),
                ConvBNAct(c4, c4, 3, 2),
                C3k2(c4, c4, n=n6, c3k=True, e=0.5, shortcut=True),
                ConvBNAct(c4, c5, 3, 2),
                C3k2(c5, c5, n=n8, c3k=True, e=0.5, shortcut=True),
                C2PSA(c5, c5, n=n9, e=0.5),
                ClassifyHead(c5, class_num=class_num),
            ]
        )
        self.stage_names = {
            0: "stem1",
            1: "stem2",
            2: "stage2",
            3: "down3",
            4: "stage3",
            5: "down4",
            6: "stage4",
            7: "down5",
            8: "stage5",
            9: "feat",
        }

    def forward_features(self, x, return_intermediates=False):
        outs = {}
        for idx, layer in enumerate(self.model[:-1]):
            x = layer(x)
            outs[self.stage_names[idx]] = x
            outs[f"model.{idx}"] = x
        if return_intermediates:
            return x, outs
        return x

    def forward_head(self, x):
        return self.model[-1](x)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)


def YOLO11_cls_n(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO11CLS(scale="n", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO11_cls_n"), use_ssld)
    return model


def YOLO11_cls_s(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO11CLS(scale="s", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO11_cls_s"), use_ssld)
    return model


def YOLO11_cls_m(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO11CLS(scale="m", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO11_cls_m"), use_ssld)
    return model


def YOLO11_cls_l(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO11CLS(scale="l", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO11_cls_l"), use_ssld)
    return model


def YOLO11_cls_x(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO11CLS(scale="x", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO11_cls_x"), use_ssld)
    return model
