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
    "YOLO26n": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/yolo26n-cls.pdparams",
    "YOLO26s": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/yolo26s-cls.pdparams",
    "YOLO26m": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/yolo26m-cls.pdparams",
    "YOLO26l": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/yolo26l-cls.pdparams",
    "YOLO26x": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/yolo26x-cls.pdparams",
}

__all__ = list(MODEL_URLS.keys())

YOLO26_CLS_CFG = {
    "n": {"depth": 0.50, "width": 0.25, "max_channels": 1024},
    "s": {"depth": 0.50, "width": 0.50, "max_channels": 1024},
    "m": {"depth": 0.50, "width": 1.00, "max_channels": 512},
    "l": {"depth": 1.00, "width": 1.00, "max_channels": 512},
    "x": {"depth": 1.00, "width": 1.50, "max_channels": 512},
}


def _load_pretrained(pretrained, model, model_url=None, use_ssld=False):
    if pretrained is False:
        return
    if pretrained is True:
        if not model_url:
            raise ValueError(
                f"No pretrained weights for {model.__class__.__name__}. "
                "Pass a local path instead."
            )
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
        return
    if isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
        return
    raise RuntimeError("pretrained must be bool or str.")


def make_divisible(x, divisor=8):
    return int(math.ceil(float(x) / divisor) * divisor)


def scale_channels(channels, width, max_channels, divisor=8):
    return make_divisible(min(channels, max_channels) * width, divisor)


def scale_depth(repeats, depth):
    return max(round(repeats * depth), 1) if repeats > 1 else repeats


class ConvBNAct(nn.Layer):
    default_act = nn.Silu()

    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, groups=1, dilation=1, act=True):
        super().__init__()
        if padding is None:
            padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels, out_channels, kernel_size, stride, padding,
            groups=groups, dilation=dilation, bias_attr=False)
        self.bn = nn.BatchNorm2D(out_channels, epsilon=1e-5, momentum=0.9)
        self.act = self.default_act if act is True else (act if isinstance(act, nn.Layer) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Layer):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, kernel_size=k[0])
        self.cv2 = ConvBNAct(c_, c2, kernel_size=k[1], groups=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3k(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvBNAct(c1, c_, 1)
        self.cv2 = ConvBNAct(c1, c_, 1)
        self.cv3 = ConvBNAct(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(paddle.concat([self.m(self.cv1(x)), self.cv2(x)], axis=1))


class C2f(nn.Layer):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1)
        self.cv2 = ConvBNAct((2 + n) * self.c, c2, 1)
        self.m = nn.LayerList(
            [Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0)
             for _ in range(n)])

    def forward(self, x):
        y = list(paddle.split(self.cv1(x), [self.c, self.c], axis=1))
        for block in self.m:
            y.append(block(y[-1]))
        return self.cv2(paddle.concat(y, axis=1))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n=n, shortcut=shortcut, g=g, e=e)
        self.m = nn.LayerList([
            C3k(self.c, self.c, n=2, shortcut=shortcut, g=g)
            if c3k
            else Bottleneck(self.c, self.c, shortcut=shortcut, g=g)
            for _ in range(n)
        ])


class Attention(nn.Layer):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim ** -0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = ConvBNAct(dim, h, 1, act=False)
        self.proj = ConvBNAct(dim, dim, 1, act=False)
        self.pe = ConvBNAct(dim, dim, 3, 1, groups=dim, act=False)

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w
        qkv = self.qkv(x).reshape(
            [b, self.num_heads, self.key_dim * 2 + self.head_dim, n])
        q, k, v = paddle.split(
            qkv, [self.key_dim, self.key_dim, self.head_dim], axis=2)
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
            ConvBNAct(c, c * 2, 1), ConvBNAct(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Layer):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2, f"C2PSA expects c1==c2, got {c1} and {c2}"
        self.c = int(c1 * e)
        self.cv1 = ConvBNAct(c1, 2 * self.c, 1)
        self.cv2 = ConvBNAct(2 * self.c, c1, 1)
        self.m = nn.Sequential(*[
            PSABlock(self.c, attn_ratio=0.5, num_heads=max(self.c // 64, 1))
            for _ in range(n)])

    def forward(self, x):
        a, b = paddle.split(self.cv1(x), [self.c, self.c], axis=1)
        b = self.m(b)
        return self.cv2(paddle.concat([a, b], axis=1))


class ClassifyHead(nn.Layer):
    """与 ultralytics Classify 完全对齐：conv(→1280) + pool + dropout + linear(→nc)"""

    def __init__(self, in_channels, class_num=1000):
        super().__init__()
        self.conv = ConvBNAct(in_channels, 1280, 1)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.drop = nn.Dropout(p=0.0)
        # 属性名必须是 linear，与 ultralytics model.10.linear 保持一致
        self.linear = nn.Linear(1280, class_num)

    def forward(self, x):
        x = self.pool(self.conv(x))
        x = paddle.flatten(x, start_axis=1)
        x = self.drop(x)
        return self.linear(x)


class YOLO26CLS(nn.Layer):
    """YOLO26 classification model, aligned with ultralytics layer indices."""

    def __init__(self, scale="n", class_num=1000):
        super().__init__()
        if scale not in YOLO26_CLS_CFG:
            raise ValueError(f"Unsupported scale: {scale}")
        cfg = YOLO26_CLS_CFG[scale]
        depth, width, max_ch = cfg["depth"], cfg["width"], cfg["max_channels"]

        def ch(c): return scale_channels(c, width, max_ch)
        def nd(n): return scale_depth(n, depth)

        use_c3k_early = scale in {"m", "l", "x"}

        # model.0 ~ model.10，与 ultralytics 层索引完全一致
        self.model = nn.LayerList([
            ConvBNAct(3, ch(64), 3, 2),                                      # 0
            ConvBNAct(ch(64), ch(128), 3, 2),                                # 1
            C3k2(ch(128), ch(256), n=nd(2), c3k=use_c3k_early, e=0.25),     # 2
            ConvBNAct(ch(256), ch(256), 3, 2),                               # 3
            C3k2(ch(256), ch(512), n=nd(2), c3k=use_c3k_early, e=0.25),     # 4
            ConvBNAct(ch(512), ch(512), 3, 2),                               # 5
            C3k2(ch(512), ch(512), n=nd(2), c3k=True),                      # 6
            ConvBNAct(ch(512), ch(1024), 3, 2),                              # 7
            C3k2(ch(1024), ch(1024), n=nd(2), c3k=True),                    # 8
            C2PSA(ch(1024), ch(1024), n=nd(2)),                             # 9
            ClassifyHead(ch(1024), class_num=class_num),                     # 10
        ])
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


def YOLO26n(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO26CLS(scale="n", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO26n"), use_ssld)
    return model


def YOLO26s(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO26CLS(scale="s", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO26s"), use_ssld)
    return model


def YOLO26m(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO26CLS(scale="m", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO26m"), use_ssld)
    return model


def YOLO26l(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO26CLS(scale="l", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO26l"), use_ssld)
    return model


def YOLO26x(pretrained=False, use_ssld=False, **kwargs):
    model = YOLO26CLS(scale="x", **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS.get("YOLO26x"), use_ssld)
    return model
