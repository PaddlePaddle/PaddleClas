# -*- coding: utf-8 -*-
from typing import Dict, Optional, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor, nn
import paddle.nn.functional as F

from mobileclip.modules.common.transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)
from mobileclip.modules.image.image_projection import GlobalPool2D
from mobileclip import logger


class ConvNormAct(nn.Layer):
    def __init__(
        self,
        cfg: Dict,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]],
        stride: Union[int, Tuple[int, ...]] = 1,
        dilation: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...]]] = None,
        groups: int = 1,
        bias: bool = False,
        padding_mode: str = "zeros",
        use_norm: bool = True,
        use_act: bool = True,
        norm_layer: Optional[nn.Layer] = None,
        act_layer: Optional[nn.Layer] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.ndim = 2

        if norm_layer is None and use_norm:
            norm_type = cfg.get("normalization", "batch_norm")
            if norm_type == "batch_norm":
                norm_layer = nn.BatchNorm2D(
                    num_features=out_channels,
                    momentum=1.0 - cfg.get("momentum", 0.1), # Paddle momentum is different
                )
            else:
                norm_layer = get_normalization_layer(
                    num_features=out_channels, norm_type=norm_type
                )

        if act_layer is None and use_act:
            act_layer = nn.GELU()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.ndim
        if isinstance(stride, int):
            stride = (stride,) * self.ndim
        if isinstance(dilation, int):
            dilation = (dilation,) * self.ndim

        if padding is None:
            padding = [int((k - 1) / 2 * d) for k, d in zip(kernel_size, dilation)]

        self.block = nn.Sequential()
        self.block.add_sublayer(
            "conv",
            nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=bias,
            ),
        )

        if use_norm:
            self.block.add_sublayer("norm", norm_layer)
        if use_act:
            self.block.add_sublayer("act", act_layer)

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class VisionTransformer(nn.Layer):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__()
        image_channels = 3
        
        self.projection_dim = kwargs.get("projection_dim", None)

        kernel_sizes_conv_stem = [4, 2, 2]
        strides_conv_stem = [4, 2, 2]
        num_embeddings = (224 // 16) ** 2

        embed_dim = cfg["embed_dim"]
        self.embed_dim = embed_dim # 保存 embed_dim 供外部查询
        n_transformer_layers = cfg["n_transformer_layers"]
        num_heads = cfg["n_attn_heads"]
        norm_layer = cfg.get("norm_layer", "layer_norm")

        conv_stem_proj_dim = max(32, embed_dim // 4)
        self.patch_embed = nn.Sequential(
            ConvNormAct(cfg, image_channels, conv_stem_proj_dim, kernel_sizes_conv_stem[0], strides_conv_stem[0], bias=False),
            ConvNormAct(cfg, conv_stem_proj_dim, conv_stem_proj_dim, kernel_sizes_conv_stem[1], strides_conv_stem[1], bias=False),
            ConvNormAct(cfg, conv_stem_proj_dim, embed_dim, kernel_sizes_conv_stem[2], strides_conv_stem[2], bias=False),
        )

        self.pos_embed = PositionalEmbedding(num_embeddings=num_embeddings, embedding_dim=embed_dim)
        
        self.transformer = nn.LayerList([
            TransformerEncoder(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_latent_dim=embed_dim * 4,
                transformer_norm_layer=norm_layer,
            ) for _ in range(n_transformer_layers)
        ])

        self.final_norm = get_normalization_layer(norm_layer, embed_dim)
        self.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = x.flatten(2).transpose([0, 2, 1])
        x = x + self.pos_embed(x.shape[1])
        for layer in self.transformer:
            x = layer(x)
        x = self.final_norm(x)
        # B 模型 (ViT) 经过池化后变为 2D，但外层 GlobalPool2D 期望 4D
        # 我们可以将其 reshape 回 [B, C, 1, 1] 以兼容池化层
        x = x.mean(axis=1) # [B, 768]
        x = x.unsqueeze(-1).unsqueeze(-1) # [B, 768, 1, 1]
        x = self.head(x)
        return x

def vit_b16(pretrained=False, **kwargs):
    cfg = {
        "embed_dim": 768,
        "n_transformer_layers": 12,
        "n_attn_heads": 12,
    }
    model = VisionTransformer(cfg, **kwargs)
    return model

def vit_l14(pretrained=False, **kwargs):
    cfg = {
        "embed_dim": 1024,
        "n_transformer_layers": 24,
        "n_attn_heads": 16,
    }
    model = VisionTransformer(cfg, **kwargs)
    return model
