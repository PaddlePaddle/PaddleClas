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
#
# reference: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/efficientvim.py
# reference: https://github.com/mlvlab/EfficientViM

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "EfficientViM_M1": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m1.e300_in1k.pdparams",
    "EfficientViM_M2": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m2.e300_in1k.pdparams",
    "EfficientViM_M3": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m3.e300_in1k.pdparams",
    "EfficientViM_M4": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m4.e300_in1k.pdparams",
    "EfficientViM_M1_Dist": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m1_dist.in1k.pdparams",
    "EfficientViM_M2_Dist": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m2_dist.in1k.pdparams",
    "EfficientViM_M3_Dist": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m3_dist.in1k.pdparams",
    "EfficientViM_M4_Dist": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvim_m4_dist.in1k.pdparams",
}


def trunc_normal_(tensor, mean=0.0, std=1.0):
    nn.initializer.TruncatedNormal(mean=mean, std=std)(tensor)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    return (x / keep_prob) * random_tensor


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def calculate_drop_path_rates(drop_path_rate, depth):
    return [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]


def make_divisible(v, divisor=8, round_limit=0.0):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class LayerNorm2d(nn.Layer):
    """LayerNorm over the channel dimension of an NCHW tensor."""

    def __init__(self, num_channels, epsilon=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[num_channels], default_initializer=nn.initializer.Constant(1.0)
        )
        self.bias = self.create_parameter(
            shape=[num_channels], default_initializer=nn.initializer.Constant(0.0)
        )
        self.epsilon = epsilon
        self.num_channels = num_channels

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x = (x - mean) / paddle.sqrt(var + self.epsilon)
        x = x * self.weight + self.bias
        return x.transpose([0, 3, 1, 2])


class SqueezeExcite(nn.Layer):
    def __init__(self, channels, rd_ratio=1.0 / 16, rd_channels=None, rd_divisor=8):
        super().__init__()
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, 0.0)
        self.fc1 = nn.Conv2D(channels, rd_channels, kernel_size=1, bias_attr=True)
        self.act = nn.ReLU()
        self.fc2 = nn.Conv2D(rd_channels, channels, kernel_size=1, bias_attr=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = paddle.mean(x, axis=[2, 3], keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(x_se)
        x_se = self.fc2(x_se)
        x_se = self.gate(x_se)
        return x * x_se


class ConvLayer1d(nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm1D,
        act_layer=nn.ReLU,
        bn_weight_init=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv1D(
            in_dim,
            out_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
        )
        self.norm = norm(out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm1D):
            self.norm.weight.set_value(
                paddle.full_like(self.norm.weight, bn_weight_init)
            )
            self.norm.bias.set_value(paddle.zeros_like(self.norm.bias))

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvLayer2d(nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        norm=nn.BatchNorm2D,
        act_layer=nn.ReLU,
        bn_weight_init=1.0,
    ):
        super().__init__()
        self.conv = nn.Conv2D(
            in_dim,
            out_dim,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
        )
        self.norm = norm(out_dim) if norm else None
        self.act = act_layer() if act_layer else None
        if self.norm is not None and isinstance(self.norm, nn.BatchNorm2D):
            self.norm.weight.set_value(
                paddle.full_like(self.norm.weight, bn_weight_init)
            )
            self.norm.bias.set_value(paddle.zeros_like(self.norm.bias))

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class Stem(nn.Layer):
    """4-layer stride-2 conv stem, reduction 16."""

    def __init__(self, in_dim, dim):
        super().__init__()
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, dim // 8, kernel_size=3, stride=2, padding=1),
            ConvLayer2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1),
            ConvLayer2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1),
            ConvLayer2d(
                dim // 2, dim, kernel_size=3, stride=2, padding=1, act_layer=None
            ),
        )

    def forward(self, x):
        return self.conv(x)


class FFN(nn.Layer):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.fc1 = ConvLayer2d(in_dim, dim, 1)
        self.fc2 = ConvLayer2d(dim, in_dim, 1, act_layer=None, bn_weight_init=0)

    def forward(self, x):
        return self.fc2(self.fc1(x))


class PatchMerging(nn.Layer):
    """MBConv-style downsample with SE and depthwise residual branches."""

    def __init__(self, in_dim, out_dim, ratio=4.0):
        super().__init__()
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, hidden_dim, kernel_size=1),
            ConvLayer2d(
                hidden_dim,
                hidden_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=hidden_dim,
            ),
            SqueezeExcite(hidden_dim, 0.25),
            ConvLayer2d(hidden_dim, out_dim, kernel_size=1, act_layer=None),
        )
        self.dwconv1 = ConvLayer2d(
            in_dim, in_dim, 3, padding=1, groups=in_dim, act_layer=None
        )
        self.dwconv2 = ConvLayer2d(
            out_dim, out_dim, 3, padding=1, groups=out_dim, act_layer=None
        )

    def forward(self, x):
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Layer):
    """Hidden State Mixer based State Space Duality (attention-like dual form)."""

    def __init__(self, dim, ssd_expand=1.0, state_dim=64, a_init_range=(1.0, 16.0)):
        super().__init__()
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * dim)
        self.state_dim = state_dim

        self.BCdt_proj = ConvLayer1d(dim, 3 * state_dim, 1, norm=None, act_layer=None)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2d(
            conv_dim,
            conv_dim,
            3,
            1,
            1,
            groups=conv_dim,
            norm=None,
            act_layer=None,
            bn_weight_init=0,
        )
        self.hz_proj = ConvLayer1d(dim, 2 * self.d_inner, 1, norm=None, act_layer=None)
        self.out_proj = ConvLayer1d(
            self.d_inner, dim, 1, norm=None, act_layer=None, bn_weight_init=0
        )

        self.a_init_range = a_init_range
        self.A = self.create_parameter(
            shape=[self.state_dim],
            default_initializer=nn.initializer.Uniform(
                low=a_init_range[0], high=a_init_range[1]
            ),
        )
        self.act = nn.SiLU()
        self.D = self.create_parameter(
            shape=[1], default_initializer=nn.initializer.Constant(1.0)
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.initializer.Uniform(low=self.a_init_range[0], high=self.a_init_range[1])(
            self.A
        )
        nn.initializer.Constant(1.0)(self.D)

    def forward(self, x):
        batch, _, height, width = x.shape
        x = x.flatten(2)

        BCdt = self.dw(self.BCdt_proj(x).reshape([batch, -1, height, width])).flatten(2)
        b, c, dt = paddle.split(BCdt, [self.state_dim] * 3, axis=1)
        a = F.softmax(dt + self.A.reshape([1, -1, 1]), axis=-1)

        ab = a * b
        h = x @ ab.transpose([0, 2, 1])

        h, z = paddle.split(self.hz_proj(h), [self.d_inner] * 2, axis=1)
        h = self.out_proj(h * self.act(z) + h * self.D)
        y = h @ c

        y = y.reshape([batch, -1, height, width])
        return y, h


class EfficientViMBlock(nn.Layer):
    """DWConv -> HSM-SSD -> DWConv -> FFN, each with a LayerScale residual."""

    def __init__(self, dim, mlp_ratio=4.0, ssd_expand=1.0, state_dim=64, drop_path=0.0):
        super().__init__()
        self.mixer = HSMSSD(dim, ssd_expand=ssd_expand, state_dim=state_dim)
        self.norm = LayerNorm2d(dim, epsilon=1e-5)

        self.dwconv1 = ConvLayer2d(
            dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None
        )
        self.dwconv2 = ConvLayer2d(
            dim, dim, 3, padding=1, groups=dim, bn_weight_init=0, act_layer=None
        )
        self.ffn = FFN(dim, int(dim * mlp_ratio))

        self.alpha = self.create_parameter(
            shape=[4, dim], default_initializer=nn.initializer.Constant(1e-4)
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def reset_parameters(self):
        nn.initializer.Constant(1e-4)(self.alpha)

    def forward(self, x):
        alpha = F.sigmoid(self.alpha).reshape([4, -1, 1, 1])

        x = x + self.drop_path(alpha[0] * (self.dwconv1(x) - x))

        x_prev = x
        x, h = self.mixer(self.norm(x))
        x = x_prev + self.drop_path(alpha[1] * (x - x_prev))

        x = x + self.drop_path(alpha[2] * (self.dwconv2(x) - x))

        x = x + self.drop_path(alpha[3] * (self.ffn(x) - x))
        return x, h


class EfficientViMStage(nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        depth,
        mlp_ratio=4.0,
        downsample=None,
        ssd_expand=1.0,
        state_dim=64,
        drop_path=None,
    ):
        super().__init__()
        assert depth > 0
        self.depth = depth
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.LayerList(
            [
                EfficientViMBlock(
                    dim=in_dim,
                    mlp_ratio=mlp_ratio,
                    ssd_expand=ssd_expand,
                    state_dim=state_dim,
                    drop_path=drop_path[i],
                )
                for i in range(depth)
            ]
        )
        self.downsample = (
            downsample(in_dim=in_dim, out_dim=out_dim)
            if downsample is not None
            else None
        )

    def forward(self, x):
        h = None
        for blk in self.blocks:
            x, h = blk(x)
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x, h


class EfficientViM(TheseusLayer):
    """EfficientViM backbone with multi-stage hidden-state fused classification head."""

    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        embed_dim=(128, 256, 512),
        depths=(2, 2, 2),
        mlp_ratio=4.0,
        ssd_expand=1.0,
        state_dim=(49, 25, 9),
        drop_rate=0.0,
        drop_path_rate=0.0,
    ):
        super().__init__()
        assert len(depths) == len(embed_dim) == len(state_dim)
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = tuple(embed_dim)

        self.patch_embed = Stem(in_dim=in_chans, dim=embed_dim[0])
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        stages = []
        block_idx = 0
        for i_layer in range(self.num_layers):
            stage_dpr = dpr[block_idx : block_idx + depths[i_layer]]
            block_idx += depths[i_layer]
            stages.append(
                EfficientViMStage(
                    in_dim=int(embed_dim[i_layer]),
                    out_dim=(
                        int(embed_dim[i_layer + 1])
                        if (i_layer < self.num_layers - 1)
                        else 0
                    ),
                    depth=depths[i_layer],
                    mlp_ratio=mlp_ratio,
                    downsample=(
                        PatchMerging if (i_layer < self.num_layers - 1) else None
                    ),
                    ssd_expand=ssd_expand,
                    state_dim=state_dim[i_layer],
                    drop_path=stage_dpr,
                )
            )
        self.stages = nn.LayerList(stages)

        self.num_features = self.head_hidden_size = sum(embed_dim) + embed_dim[-1]
        self.weights = self.create_parameter(
            shape=[self.num_layers + 1],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.norm = nn.LayerList(
            [
                *[nn.LayerNorm(dim, epsilon=1e-5) for dim in embed_dim],
                LayerNorm2d(embed_dim[-1], epsilon=1e-5),
            ]
        )
        self.head_drop = nn.Dropout(drop_rate)
        self.heads = self._build_heads(num_classes)
        self.init_weights(needs_reset=False)

    def _build_heads(self, num_classes):
        return nn.LayerList(
            [
                nn.Linear(dim, num_classes)
                for dim in (*self.embed_dim, self.embed_dim[-1])
            ]
        )

    def init_weights(self, needs_reset=True):
        nn.initializer.Constant(1.0)(self.weights)
        if hasattr(self, "weights_dist"):
            nn.initializer.Constant(1.0)(self.weights_dist)
        self.apply(lambda m: self._init_weights(m, needs_reset=needs_reset))

    def _init_weights(self, m, needs_reset=True):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.initializer.Constant(0.0)(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D, nn.BatchNorm1D)):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        elif needs_reset and hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def forward_features(self, x):
        x = self.patch_embed(x)
        features = []
        for stage, norm in zip(self.stages, self.norm):
            _, x, h = stage(x)
            features.append(norm(h.transpose([0, 2, 1])).transpose([0, 2, 1]))
        features.append(self.norm[-1](x).flatten(2))
        return features

    def forward_head(self, x):
        x = [self.head_drop(t.mean(axis=-1)) for t in x]
        if self.num_classes == 0:
            return paddle.concat(x, axis=1)
        weights = F.softmax(self.weights, axis=-1)
        logits = [weights[i] * head(x[i]) for i, head in enumerate(self.heads)]
        return paddle.stack(logits).sum(axis=0)

    def forward(self, x):
        x = self.forward_features(x)
        return self.forward_head(x)


class EfficientViMDistilled(EfficientViM):
    """EfficientViM with a parallel distilled head, averaged with the primary head in eval."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_classes = self.num_classes
        self.weights_dist = self.create_parameter(
            shape=[self.num_layers + 1],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.heads_dist = self._build_heads(num_classes)
        self.heads_dist.apply(self._init_weights)
        self.distilled_training = False

    def forward_head(self, x):
        x = [self.head_drop(t.mean(axis=-1)) for t in x]
        if self.num_classes == 0:
            return paddle.concat(x, axis=1)
        weights = F.softmax(self.weights, axis=-1)
        weights_dist = F.softmax(self.weights_dist, axis=-1)
        logits = [weights[i] * head(x[i]) for i, head in enumerate(self.heads)]
        logits_dist = [
            weights_dist[i] * head(x[i]) for i, head in enumerate(self.heads_dist)
        ]
        z = paddle.stack(logits).sum(axis=0)
        z_dist = paddle.stack(logits_dist).sum(axis=0)
        if self.distilled_training and self.training:
            return z, z_dist
        return (z + z_dist) / 2


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


def _efficientvim(class_num=1000, distilled=False, **kwargs):
    Model = EfficientViMDistilled if distilled else EfficientViM
    return Model(num_classes=class_num, **kwargs)


def EfficientViM_M1(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(128, 192, 320), depths=(2, 2, 2), state_dim=(49, 25, 9), **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M1"], use_ssld)
    return model


def EfficientViM_M2(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(128, 256, 512), depths=(2, 2, 2), state_dim=(49, 25, 9), **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M2"], use_ssld)
    return model


def EfficientViM_M3(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(224, 320, 512), depths=(2, 2, 2), state_dim=(49, 25, 9), **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M3"], use_ssld)
    return model


def EfficientViM_M4(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(224, 320, 512), depths=(3, 4, 2), state_dim=(64, 32, 16), **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M4"], use_ssld)
    return model


def EfficientViM_M1_Dist(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(128, 192, 320),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
        distilled=True,
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M1_Dist"], use_ssld)
    return model


def EfficientViM_M2_Dist(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(128, 256, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
        distilled=True,
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M2_Dist"], use_ssld)
    return model


def EfficientViM_M3_Dist(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(224, 320, 512),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
        distilled=True,
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M3_Dist"], use_ssld)
    return model


def EfficientViM_M4_Dist(pretrained=False, use_ssld=False, **kwargs):
    model = _efficientvim(
        embed_dim=(224, 320, 512),
        depths=(3, 4, 2),
        state_dim=(64, 32, 16),
        distilled=True,
        **kwargs,
    )
    _load_pretrained(pretrained, model, MODEL_URLS["EfficientViM_M4_Dist"], use_ssld)
    return model
