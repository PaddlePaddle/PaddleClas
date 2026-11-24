# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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
#
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Adapted for Paddle from apple-aiml-research/ml-mobileclip.
# See docs/zh_CN/models/ImageNet1k/MobileCLIP/licenses for licenses.
"""MobileCLIP v1 image and text encoders."""

import math

import paddle
from paddle import nn
from paddle.nn import functional as F

from ..base.theseus_layer import TheseusLayer
from .mixnet import SEBlock
from .repvgg import ConvBN
from .vision_transformer import Attention, Mlp
from ....utils.save_load import load_dygraph_pretrain

__all__ = [
    "MobileCLIP", "MobileCLIP_S0", "MobileCLIP_S1", "MobileCLIP_S2",
    "MobileCLIP_B", "MobileCLIP_B_LT"
]

_VARIANTS = {
    "mobileclip_s0": ([2, 6, 10, 2], [64, 128, 256, 512], 4, 256),
    "mobileclip_s1": ([4, 12, 20, 4], [64, 128, 256, 512], 12, 256),
    "mobileclip_s2": ([4, 12, 24, 4], [80, 160, 320, 640], 12, 256),
    "mobileclip_b": (None, None, 12, 224),
    "mobileclip_blt": (None, None, 12, 224),
}


def _parameter(layer, shape, value=None, std=0.02):
    initializer = (nn.initializer.TruncatedNormal(
        std=std) if value is None else nn.initializer.Constant(value))
    return layer.create_parameter(shape, default_initializer=initializer)


class _FusedConv(nn.Layer):
    def __init__(self,
                 in_dim,
                 out_dim,
                 kernel,
                 stride=1,
                 groups=1,
                 se_ratio=None,
                 activation=True):
        super().__init__()
        padding = (tuple(
            k // 2
            for k in kernel) if isinstance(kernel, tuple) else kernel // 2)
        self.conv = nn.Conv2D(
            in_dim, out_dim, kernel, stride, padding, groups=groups)
        self.se = (SEBlock(out_dim, mid_channels=int(out_dim * se_ratio))
                   if se_ratio else nn.Identity())
        self.activation = nn.GELU() if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        return self.activation(self.se(x))


class _ConvFFN(nn.Layer):
    def __init__(self, dim, ratio=3, text=False):
        super().__init__()
        kernel, padding = ((1, 11), (0, 5)) if text else (7, 3)
        self.conv = ConvBN(
            dim, dim, kernel, stride=1, padding=padding, groups=dim)
        self.fc1 = nn.Conv2D(dim, dim * ratio, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2D(dim * ratio, dim, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(self.conv(x))))


class _SpatialAttention(Attention):
    def __init__(self, dim):
        super().__init__(dim, num_heads=dim // 32)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose([0, 2, 1])
        x = super().forward(x)
        return x.transpose([0, 2, 1]).reshape([b, c, h, w])


class _MixerBlock(nn.Layer):
    def __init__(self, dim, text=False, attention=False):
        super().__init__()
        self.text = text
        self.attention = attention
        if attention:
            self.norm = nn.BatchNorm2D(dim, momentum=0.9, epsilon=1e-5)
            self.token_mixer = _SpatialAttention(dim)
            self.layer_scale_1 = _parameter(self, [dim, 1, 1], 1e-5)
            self.layer_scale_2 = _parameter(self, [dim, 1, 1], 1e-5)
        else:
            kernel = (1, 11) if text else 3
            self.token_mixer = _FusedConv(
                dim, dim, kernel, groups=dim, activation=False)
            self.layer_scale = _parameter(self, [dim, 1, 1], 1e-5)
        self.convffn = _ConvFFN(dim, ratio=4 if text else 3, text=text)

    def forward(self, x, **kwargs):
        if self.text:
            x = x.transpose([0, 2, 1]).unsqueeze(2)
        if self.attention:
            x = x + self.layer_scale_1 * self.token_mixer(self.norm(x))
            x = x + self.layer_scale_2 * self.convffn(x)
        else:
            x = self.token_mixer(x)
            x = x + self.layer_scale * self.convffn(x)
        return x.squeeze(2).transpose([0, 2, 1]) if self.text else x


class _PatchEmbed(nn.Layer):
    def __init__(self, in_dim, out_dim, use_se):
        super().__init__()
        self.proj = nn.Sequential(
            _FusedConv(
                in_dim,
                out_dim,
                7,
                stride=2,
                groups=in_dim,
                se_ratio=0.25 if use_se else None),
            _FusedConv(out_dim, out_dim, 1))

    def forward(self, x):
        return self.proj(x)


class _Projection(nn.Layer):
    def __init__(self, dim, pool=False):
        super().__init__()
        self.proj = _parameter(self, [dim, 512], std=dim**-0.5)
        self.pool = pool

    def forward(self, x):
        if self.pool:
            x = x.mean(axis=[2, 3])
        return x @ self.proj


class _MCi(nn.Layer):
    def __init__(self, layers, dims):
        super().__init__()
        self.patch_embed = nn.Sequential(
            _FusedConv(3, dims[0], 3, stride=2),
            _FusedConv(dims[0], dims[0], 3, stride=2, groups=dims[0]),
            _FusedConv(dims[0], dims[0], 1))
        network = []
        for i, (depth, dim) in enumerate(zip(layers, dims)):
            if i == 3:
                network.append(
                    _FusedConv(dim, dim, 7, groups=dim, activation=False))
            network.append(
                nn.Sequential(
                    *
                    [_MixerBlock(dim, attention=i == 3)
                     for _ in range(depth)]))
            if i < 3:
                network.append(_PatchEmbed(dim, dims[i + 1], use_se=i >= 1))
        self.network = nn.LayerList(network)
        self.conv_exp = _FusedConv(
            dims[-1], dims[-1] * 2, 3, groups=dims[-1], se_ratio=0.0625)
        self.head = _Projection(dims[-1] * 2, pool=True)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.network:
            x = block(x)
        return self.head(self.conv_exp(x))


class _LearnablePosition(nn.Layer):
    def __init__(self, length, dim):
        super().__init__()
        self.pos_embed = _parameter(self, [1, 1, length, dim], std=dim**-0.5)
        self.length = length
        self.dim = dim

    def forward(self, length):
        position = self.pos_embed
        if length != self.length:
            position = F.interpolate(
                position,
                size=[length, self.dim],
                mode="bilinear",
                align_corners=False,
                align_mode=0)
        return position.reshape([1, length, self.dim])


class _Attention(nn.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.heads = heads
        self.scale = (dim // heads)**-0.5

    def forward(self, x, attn_mask=None):
        b, length, dim = x.shape
        qkv = self.qkv_proj(x).reshape([b, length, 3, self.heads, -1])
        qkv = qkv.transpose([0, 3, 2, 1, 4])
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        attn = (q * self.scale) @ k.transpose([0, 1, 3, 2])
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn.astype("float32"), axis=-1).astype(x.dtype)
        out = (attn @ v).transpose([0, 2, 1, 3]).reshape([b, length, dim])
        return self.out_proj(out)


class _Transformer(nn.Layer):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-5)
        self.attn = _Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-5)
        self.mlp = Mlp(dim, hidden_features=dim * 4)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask)
        return x + self.mlp(self.norm2(x))


class _ConvStem(nn.Layer):
    def __init__(self, in_dim, out_dim, kernel, norm=True):
        super().__init__()
        modules = [("conv",
                    nn.Conv2D(
                        in_dim,
                        out_dim,
                        kernel,
                        stride=kernel,
                        padding=(kernel - 1) // 2,
                        bias_attr=not norm))]
        if norm:
            modules.extend([("norm",
                             nn.BatchNorm2D(
                                 out_dim, momentum=0.9, epsilon=1e-5)),
                            ("act", nn.GELU())])
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        return self.block(x)


class _ViT(nn.Layer):
    def __init__(self):
        super().__init__()
        self.patch_emb = nn.Sequential(
            _ConvStem(3, 192, 4), _ConvStem(192, 192, 2),
            _ConvStem(192, 768, 2, norm=False))
        self.cls_token = _parameter(self, [1, 1, 768])
        self.pos_embed = _LearnablePosition(196, 768)
        self.transformer = nn.Sequential(
            *[_Transformer(768, 12) for _ in range(12)])
        self.post_transformer_norm = nn.LayerNorm(768, epsilon=1e-5)
        self.classifier = _Projection(768)

    def forward(self, x):
        x = self.patch_emb(x).flatten(2).transpose([0, 2, 1])
        x = x + self.pos_embed(x.shape[1])
        x = paddle.concat([self.cls_token.expand([x.shape[0], -1, -1]), x],
                          axis=1)
        x = self.post_transformer_norm(self.transformer(x))
        return self.classifier(x[:, 0])


def _create_image_encoder(variant):
    layers, dims, _, _ = _VARIANTS[variant]
    return _MCi(layers, dims) if layers is not None else _ViT()


class _TextEncoder(nn.Layer):
    def __init__(self, variant):
        super().__init__()
        self.embedding_layer = nn.Embedding(49408, 512)
        self.positional_embedding = _LearnablePosition(77, 512)
        layers = [_Transformer(512, 8) for _ in range(_VARIANTS[variant][2])]
        if variant == "mobileclip_s0":
            layers = [_MixerBlock(512, text=True)
                      ] + layers + [_MixerBlock(512, text=True)]
        self.transformer = nn.LayerList(layers)
        self.final_layer_norm = nn.LayerNorm(512, epsilon=1e-5)
        self.projection_layer = _parameter(self, [512, 512])
        self.causal = variant in ("mobileclip_b", "mobileclip_blt")

    def forward(self, tokens):
        x = self.embedding_layer(tokens) + self.positional_embedding(
            tokens.shape[1])
        mask = None
        if self.causal:
            mask = paddle.triu(
                paddle.full([tokens.shape[1], tokens.shape[1]],
                            float("-inf"),
                            dtype=x.dtype),
                diagonal=1)
        for layer in self.transformer:
            x = layer(x, attn_mask=mask)
        x = self.final_layer_norm(x)
        indices = paddle.stack(
            [paddle.arange(tokens.shape[0]),
             tokens.argmax(axis=-1)], axis=1)
        return paddle.gather_nd(x, indices) @ self.projection_layer


class MobileCLIP(nn.Layer):
    """MobileCLIP image-text model."""

    def __init__(self, variant="mobileclip_s0", output_dict=False):
        super().__init__()
        if variant not in _VARIANTS:
            raise ValueError("Unknown MobileCLIP variant: {}".format(variant))
        self.image_encoder = _create_image_encoder(variant)
        self.text_encoder = _TextEncoder(variant)
        self.logit_scale = _parameter(self, [], math.log(1.0 / 0.07))
        self.output_dict = output_dict
        self.image_size = _VARIANTS[variant][3]

    def encode_image(self, image, normalize=False):
        x = self.image_encoder(image)
        return F.normalize(x, axis=-1, epsilon=1e-12) if normalize else x

    def encode_text(self, tokens, normalize=False):
        x = self.text_encoder(tokens)
        return F.normalize(x, axis=-1, epsilon=1e-12) if normalize else x

    def forward(self, image=None, text=None):
        image = self.encode_image(
            image, normalize=True) if image is not None else None
        text = self.encode_text(
            text, normalize=True) if text is not None else None
        scale = paddle.clip(self.logit_scale.exp(), 0, 100)
        if self.output_dict:
            return {
                "image_features": image,
                "text_features": text,
                "logit_scale": scale
            }
        return image, text, scale


class _MobileCLIPClassifier(TheseusLayer):
    def __init__(self, variant, class_num=1000):
        super().__init__()
        self.class_num = class_num
        self.num_features = 512
        self.image_encoder = _create_image_encoder(variant)
        self.head = nn.Linear(self.num_features,
                              class_num) if class_num > 0 else nn.Identity()

    def forward_features(self, x):
        return self.image_encoder(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def _classifier(variant, pretrained=False, use_ssld=False, **kwargs):
    model = _MobileCLIPClassifier(variant, **kwargs)
    if pretrained is False:
        return model
    if isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained, use_ssld=use_ssld)
    elif pretrained is True:
        raise ValueError("Pass the converted image encoder .pdparams path.")
    else:
        raise RuntimeError("pretrained must be a string or boolean.")
    return model


def MobileCLIP_S0(pretrained=False, use_ssld=False, **kwargs):
    return _classifier("mobileclip_s0", pretrained, use_ssld, **kwargs)


def MobileCLIP_S1(pretrained=False, use_ssld=False, **kwargs):
    return _classifier("mobileclip_s1", pretrained, use_ssld, **kwargs)


def MobileCLIP_S2(pretrained=False, use_ssld=False, **kwargs):
    return _classifier("mobileclip_s2", pretrained, use_ssld, **kwargs)


def MobileCLIP_B(pretrained=False, use_ssld=False, **kwargs):
    return _classifier("mobileclip_b", pretrained, use_ssld, **kwargs)


def MobileCLIP_B_LT(pretrained=False, use_ssld=False, **kwargs):
    return _classifier("mobileclip_blt", pretrained, use_ssld, **kwargs)
