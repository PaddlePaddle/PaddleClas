# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/bytedance/Next-ViT/blob/main/classification/nextvit.py
# reference: https://arxiv.org/abs/2207.05501

from functools import partial

import paddle
from paddle import nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from .vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "NextViT_small_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_small_224_pretrained.pdparams",
    "NextViT_base_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_base_224_pretrained.pdparams",
    "NextViT_large_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_large_224_pretrained.pdparams",
    "NextViT_small_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_small_384_pretrained.pdparams",
    "NextViT_base_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_base_384_pretrained.pdparams",
    "NextViT_large_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/NextViT_large_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

NORM_EPS = 1e-5


def rearrange(x, pattern, **axes_lengths):
    if 'b (h w) c -> b c h w' == pattern:
        b, n, c = x.shape
        h = axes_lengths.pop('h', -1)
        w = axes_lengths.pop('w', -1)
        h = h if w == -1 else n // w
        w = w if h == -1 else n // h
        return x.transpose([0, 2, 1]).reshape([b, c, h, w])
    if 'b c h w -> b (h w) c' == pattern:
        b, c, h, w = x.shape
        return x.reshape([b, c, h * w]).transpose([0, 2, 1])
    if 'b t (h d) -> b h t d' == pattern:
        b, t, h_d = x.shape
        h = axes_lengths['h']
        return x.reshape([b, t, h, h_d // h]).transpose([0, 2, 1, 3])
    if 'b h t d -> b t (h d)' == pattern:
        b, h, t, d = x.shape
        return x.transpose([0, 2, 1, 3]).reshape([b, t, h * d])

    raise NotImplementedError(
        "Rearrangement '{}' has not been implemented.".format(pattern))


def merge_pre_bn(layer, pre_bn_1, pre_bn_2=None):
    """ Merge pre BN to reduce inference runtime.
    """
    weight = layer.weight
    if isinstance(layer, nn.Linear):
        weight = weight.transpose([1, 0])
    bias = layer.bias
    if pre_bn_2 is None:
        scale_invstd = (pre_bn_1._variance + pre_bn_1._epsilon).pow(-0.5)
        extra_weight = scale_invstd * pre_bn_1.weight
        extra_bias = pre_bn_1.bias - pre_bn_1.weight * pre_bn_1._mean * scale_invstd
    else:
        scale_invstd_1 = (pre_bn_1._variance + pre_bn_1._epsilon).pow(-0.5)
        scale_invstd_2 = (pre_bn_2._variance + pre_bn_2._epsilon).pow(-0.5)

        extra_weight = scale_invstd_1 * pre_bn_1.weight * scale_invstd_2 * pre_bn_2.weight
        extra_bias = scale_invstd_2 * pre_bn_2.weight * (
            pre_bn_1.bias - pre_bn_1.weight * pre_bn_1._mean * scale_invstd_1 -
            pre_bn_2._mean) + pre_bn_2.bias
    if isinstance(layer, nn.Linear):
        extra_bias = weight @extra_bias

        weight = weight.multiply(
            extra_weight.reshape([1, weight.shape[1]]).expand_as(weight))
        weight = weight.transpose([1, 0])
    elif isinstance(layer, nn.Conv2D):
        assert weight.shape[2] == 1 and weight.shape[3] == 1

        weight = weight.reshape([weight.shape[0], weight.shape[1]])
        extra_bias = weight @extra_bias
        weight = weight.multiply(
            extra_weight.reshape([1, weight.shape[1]]).expand_as(weight))
        weight = weight.reshape([weight.shape[0], weight.shape[1], 1, 1])
    bias = bias.add(extra_bias)

    layer.weight.set_value(weight)
    layer.bias.set_value(bias)


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=groups,
            bias_attr=False)
        self.norm = nn.BatchNorm2D(out_channels, epsilon=NORM_EPS)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class PatchEmbed(nn.Layer):
    def __init__(self, in_channels, out_channels, stride=1):
        super(PatchEmbed, self).__init__()
        norm_layer = partial(nn.BatchNorm2D, epsilon=NORM_EPS)
        if stride == 2:
            self.avgpool = nn.AvgPool2D((2, 2), stride=2, ceil_mode=True)
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False)
            self.norm = norm_layer(out_channels)
        elif in_channels != out_channels:
            self.avgpool = nn.Identity()
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias_attr=False)
            self.norm = norm_layer(out_channels)
        else:
            self.avgpool = nn.Identity()
            self.conv = nn.Identity()
            self.norm = nn.Identity()

    def forward(self, x):
        return self.norm(self.conv(self.avgpool(x)))


class MHCA(nn.Layer):
    """
    Multi-Head Convolutional Attention
    """

    def __init__(self, out_channels, head_dim):
        super(MHCA, self).__init__()
        norm_layer = partial(nn.BatchNorm2D, epsilon=NORM_EPS)
        self.group_conv3x3 = nn.Conv2D(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=out_channels // head_dim,
            bias_attr=False)
        self.norm = norm_layer(out_channels)
        self.act = nn.ReLU()
        self.projection = nn.Conv2D(
            out_channels, out_channels, kernel_size=1, bias_attr=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 out_features=None,
                 mlp_ratio=None,
                 drop=0.,
                 bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_dim = _make_divisible(in_features * mlp_ratio, 32)
        self.conv1 = nn.Conv2D(
            in_features,
            hidden_dim,
            kernel_size=1,
            bias_attr=None if bias == True else False)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2D(
            hidden_dim,
            out_features,
            kernel_size=1,
            bias_attr=None if bias == True else False)
        self.drop = nn.Dropout(drop)

    def merge_bn(self, pre_norm):
        merge_pre_bn(self.conv1, pre_norm)
        self.is_bn_merged = True

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.drop(x)
        return x


class NCB(nn.Layer):
    """
    Next Convolution Block
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 path_dropout=0.0,
                 drop=0.0,
                 head_dim=32,
                 mlp_ratio=3):
        super(NCB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        norm_layer = partial(nn.BatchNorm2D, epsilon=NORM_EPS)
        assert out_channels % head_dim == 0

        self.patch_embed = PatchEmbed(in_channels, out_channels, stride)
        self.mhca = MHCA(out_channels, head_dim)
        self.attention_path_dropout = DropPath(path_dropout)

        self.norm = norm_layer(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop, bias=True)
        self.mlp_path_dropout = DropPath(path_dropout)
        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.mlp.merge_bn(self.norm)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.attention_path_dropout(self.mhca(x))

        if not self.is_bn_merged:
            out = self.norm(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class E_MHSA(nn.Layer):
    """
    Efficient Multi-Head Self Attention
    """

    def __init__(self,
                 dim,
                 out_dim=None,
                 head_dim=32,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop=0,
                 proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim if out_dim is not None else dim
        self.num_heads = self.dim // head_dim
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, self.dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, self.dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, self.dim, bias_attr=qkv_bias)
        self.proj = nn.Linear(self.dim, self.out_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        self.N_ratio = sr_ratio**2
        if sr_ratio > 1:
            self.sr = nn.AvgPool1D(
                kernel_size=self.N_ratio, stride=self.N_ratio)
            self.norm = nn.BatchNorm1D(dim, epsilon=NORM_EPS)
        self.is_bn_merged = False

    def merge_bn(self, pre_bn):
        merge_pre_bn(self.q, pre_bn)
        if self.sr_ratio > 1:
            merge_pre_bn(self.k, pre_bn, self.norm)
            merge_pre_bn(self.v, pre_bn, self.norm)
        else:
            merge_pre_bn(self.k, pre_bn)
            merge_pre_bn(self.v, pre_bn)
        self.is_bn_merged = True

    def forward(self, x):
        B, N, C = x.shape
        q = self.q(x)
        q = q.reshape(
            [B, N, self.num_heads, int(C // self.num_heads)]).transpose(
                [0, 2, 1, 3])
        if self.sr_ratio > 1:
            x_ = x.transpose([0, 2, 1])
            x_ = self.sr(x_)
            if not self.is_bn_merged:
                x_ = self.norm(x_)
            x_ = x_.transpose([0, 2, 1])

            k = self.k(x_)
            k = k.reshape(
                [B, k.shape[1], self.num_heads, int(C // self.num_heads)
                 ]).transpose([0, 2, 3, 1])
            v = self.v(x_)
            v = v.reshape(
                [B, v.shape[1], self.num_heads, int(C // self.num_heads)
                 ]).transpose([0, 2, 1, 3])
        else:
            k = self.k(x)
            k = k.reshape(
                [B, k.shape[1], self.num_heads, int(C // self.num_heads)
                 ]).transpose([0, 2, 3, 1])
            v = self.v(x)
            v = v.reshape(
                [B, v.shape[1], self.num_heads, int(C // self.num_heads)
                 ]).transpose([0, 2, 1, 3])
        attn = (q @k) * self.scale
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class NTB(nn.Layer):
    """
    Next Transformer Block
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            path_dropout,
            stride=1,
            sr_ratio=1,
            mlp_ratio=2,
            head_dim=32,
            mix_block_ratio=0.75,
            attn_drop=0.0,
            drop=0.0, ):
        super(NTB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_block_ratio = mix_block_ratio
        norm_func = partial(nn.BatchNorm2D, epsilon=NORM_EPS)

        self.mhsa_out_channels = _make_divisible(
            int(out_channels * mix_block_ratio), 32)
        self.mhca_out_channels = out_channels - self.mhsa_out_channels

        self.patch_embed = PatchEmbed(in_channels, self.mhsa_out_channels,
                                      stride)
        self.norm1 = norm_func(self.mhsa_out_channels)
        self.e_mhsa = E_MHSA(
            self.mhsa_out_channels,
            head_dim=head_dim,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.mhsa_path_dropout = DropPath(path_dropout * mix_block_ratio)

        self.projection = PatchEmbed(
            self.mhsa_out_channels, self.mhca_out_channels, stride=1)
        self.mhca = MHCA(self.mhca_out_channels, head_dim=head_dim)
        self.mhca_path_dropout = DropPath(path_dropout * (1 - mix_block_ratio))

        self.norm2 = norm_func(out_channels)
        self.mlp = Mlp(out_channels, mlp_ratio=mlp_ratio, drop=drop)
        self.mlp_path_dropout = DropPath(path_dropout)

        self.is_bn_merged = False

    def merge_bn(self):
        if not self.is_bn_merged:
            self.e_mhsa.merge_bn(self.norm1)
            self.mlp.merge_bn(self.norm2)
            self.is_bn_merged = True

    def forward(self, x):
        x = self.patch_embed(x)

        B, C, H, W = x.shape
        if not self.is_bn_merged:
            out = self.norm1(x)
        else:
            out = x
        out = rearrange(out, "b c h w -> b (h w) c")  # b n c
        out = self.e_mhsa(out)
        out = self.mhsa_path_dropout(out)
        x = x + rearrange(out, "b (h w) c -> b c h w", h=H)

        out = self.projection(x)
        out = out + self.mhca_path_dropout(self.mhca(out))
        x = paddle.concat([x, out], axis=1)

        if not self.is_bn_merged:
            out = self.norm2(x)
        else:
            out = x
        x = x + self.mlp_path_dropout(self.mlp(out))
        return x


class NextViT(nn.Layer):
    def __init__(self,
                 stem_chs,
                 depths,
                 path_dropout,
                 attn_drop=0,
                 drop=0,
                 class_num=1000,
                 strides=[1, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 head_dim=32,
                 mix_block_ratio=0.75):
        super(NextViT, self).__init__()

        self.stage_out_channels = [
            [96] * (depths[0]), [192] * (depths[1] - 1) + [256],
            [384, 384, 384, 384, 512] * (depths[2] // 5),
            [768] * (depths[3] - 1) + [1024]
        ]

        # Next Hybrid Strategy
        self.stage_block_types = [[NCB] * depths[0],
                                  [NCB] * (depths[1] - 1) + [NTB],
                                  [NCB, NCB, NCB, NCB, NTB] * (depths[2] // 5),
                                  [NCB] * (depths[3] - 1) + [NTB]]

        self.stem = nn.Sequential(
            ConvBNReLU(
                3, stem_chs[0], kernel_size=3, stride=2),
            ConvBNReLU(
                stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            ConvBNReLU(
                stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            ConvBNReLU(
                stem_chs[2], stem_chs[2], kernel_size=3, stride=2), )
        input_channel = stem_chs[-1]
        features = []
        idx = 0
        dpr = [
            x.item() for x in paddle.linspace(0, path_dropout, sum(depths))
        ]  # stochastic depth decay rule
        for stage_id in range(len(depths)):
            numrepeat = depths[stage_id]
            output_channels = self.stage_out_channels[stage_id]
            block_types = self.stage_block_types[stage_id]
            for block_id in range(numrepeat):
                if strides[stage_id] == 2 and block_id == 0:
                    stride = 2
                else:
                    stride = 1
                output_channel = output_channels[block_id]
                block_type = block_types[block_id]
                if block_type is NCB:
                    layer = NCB(input_channel,
                                output_channel,
                                stride=stride,
                                path_dropout=dpr[idx + block_id],
                                drop=drop,
                                head_dim=head_dim)
                    features.append(layer)
                elif block_type is NTB:
                    layer = NTB(input_channel,
                                output_channel,
                                path_dropout=dpr[idx + block_id],
                                stride=stride,
                                sr_ratio=sr_ratios[stage_id],
                                head_dim=head_dim,
                                mix_block_ratio=mix_block_ratio,
                                attn_drop=attn_drop,
                                drop=drop)
                    features.append(layer)
                input_channel = output_channel
            idx += numrepeat
        self.features = nn.Sequential(*features)

        self.norm = nn.BatchNorm2D(output_channel, epsilon=NORM_EPS)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.proj_head = nn.Sequential(nn.Linear(output_channel, class_num), )

        self.stage_out_idx = [
            sum(depths[:idx + 1]) - 1 for idx in range(len(depths))
        ]
        self._initialize_weights()

    def merge_bn(self):
        self.eval()
        for idx, layer in self.named_sublayers():
            if isinstance(layer, NCB) or isinstance(layer, NTB):
                layer.merge_bn()

    def _initialize_weights(self):
        for n, m in self.named_sublayers():
            if isinstance(m, (nn.BatchNorm2D, nn.GroupNorm, nn.LayerNorm,
                              nn.BatchNorm1D)):
                ones_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, nn.Conv2D):
                trunc_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        for layer in self.features:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        x = self.proj_head(x)
        return x


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


def NextViT_small_224(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 10, 3],
        path_dropout=0.1,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_small_224"], use_ssld=use_ssld)
    return model


def NextViT_base_224(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 20, 3],
        path_dropout=0.2,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_base_224"], use_ssld=use_ssld)
    return model


def NextViT_large_224(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 30, 3],
        path_dropout=0.2,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_large_224"], use_ssld=use_ssld)
    return model


def NextViT_small_384(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 10, 3],
        path_dropout=0.1,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_small_384"], use_ssld=use_ssld)
    return model


def NextViT_base_384(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 20, 3],
        path_dropout=0.2,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_base_384"], use_ssld=use_ssld)
    return model


def NextViT_large_384(pretrained=False, use_ssld=False, **kwargs):
    model = NextViT(
        stem_chs=[64, 32, 64],
        depths=[3, 4, 30, 3],
        path_dropout=0.2,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["NextViT_large_384"], use_ssld=use_ssld)
    return model
