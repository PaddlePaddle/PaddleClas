# -*- coding: utf-8 -*-
import copy
from functools import partial
from typing import List, Tuple, Optional, Union

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from mobileclip.modules.common.mobileone import MobileOneBlock
from mobileclip.modules.image.replknet import ReparamLargeKernelConv


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}


def convolutional_stem(
    in_channels: int, out_channels: int, inference_mode: bool = False
) -> nn.Sequential:
    return nn.Sequential(
        MobileOneBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            groups=out_channels,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
        MobileOneBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            inference_mode=inference_mode,
            use_se=False,
            num_conv_branches=1,
        ),
    )


class MHSA(nn.Layer):
    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        shape = x.shape
        B, C, H, W = shape
        N = H * W
        if len(shape) == 4:
            x = x.flatten(2).transpose([0, 2, 1])  # (B, N, C)
        
        qkv = (
            self.qkv(x)
            .reshape([B, N, 3, self.num_heads, self.head_dim])
            .transpose([2, 0, 3, 1, 4])
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose([0, 2, 1, 3]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        if len(shape) == 4:
            x = x.transpose([0, 2, 1]).reshape([B, C, H, W])

        return x


class PatchEmbed(nn.Layer):
    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            ReparamLargeKernelConv(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=stride,
                groups=in_channels,
                small_kernel=3,
                inference_mode=inference_mode,
                use_se=use_se,
            ),
            MobileOneBlock(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
            )
        )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.proj(x)
        return x


class RepMixer(nn.Layer):
    def __init__(
        self,
        dim,
        kernel_size=3,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.kernel_size // 2,
                groups=self.dim,
                bias_attr=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=dim,
                use_act=False,
            )
            self.use_layer_scale = use_layer_scale
            if use_layer_scale:
                self.layer_scale = self.create_parameter(
                    shape=[dim, 1, 1],
                    default_initializer=nn.initializer.Constant(layer_scale_init_value)
                )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if hasattr(self, "reparam_conv"):
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale * (self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            return x


class ConvFFN(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: nn.Layer = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.conv = nn.Sequential(
            ('conv', nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=7,
                padding=3,
                groups=in_channels,
                bias_attr=False,
            )),
            ('bn', nn.BatchNorm2D(num_features=out_channels))
        )
        self.fc1 = nn.Conv2D(in_channels, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_channels, out_channels, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RepCPE(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
    ) -> None:
        super(RepCPE, self).__init__()
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        
        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.groups = embed_dim

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.spatial_shape,
                stride=1,
                padding=int(self.spatial_shape[0] // 2),
                groups=self.embed_dim,
                bias_attr=True,
            )
        else:
            self.pe = nn.Conv2D(
                in_channels,
                embed_dim,
                spatial_shape,
                1,
                int(spatial_shape[0] // 2),
                bias_attr=True,
                groups=embed_dim,
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if hasattr(self, "reparam_conv"):
            x = x + self.reparam_conv(x)
        else:
            x = x + self.pe(x)
        return x


class RepMixerBlock(nn.Layer):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: nn.Layer = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
    ):
        super().__init__()
        self.token_mixer = RepMixer(
            dim,
            kernel_size=kernel_size,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
        )
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = self.create_parameter(
                shape=[dim, 1, 1],
                default_initializer=nn.initializer.Constant(layer_scale_init_value)
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.layer_scale * self.convffn(x))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path(self.convffn(x))
        return x


class AttentionBlock(nn.Layer):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: nn.Layer = nn.GELU,
        norm_layer: nn.Layer = nn.BatchNorm2D,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.token_mixer = MHSA(dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = self.create_parameter(
                shape=[dim, 1, 1],
                default_initializer=nn.initializer.Constant(layer_scale_init_value)
            )
            self.layer_scale_2 = self.create_parameter(
                shape=[dim, 1, 1],
                default_initializer=nn.initializer.Constant(layer_scale_init_value)
            )

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.layer_scale_2 * self.convffn(x))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x


class FastViT(nn.Layer):
    def __init__(
        self,
        layers,
        token_mixers: Tuple[str, ...],
        embed_dims=None,
        mlp_ratios=None,
        downsamples=None,
        se_downsamples=None,
        repmixer_kernel_size=3,
        norm_layer: nn.Layer = nn.BatchNorm2D,
        act_layer: nn.Layer = nn.GELU,
        num_classes=1000,
        pos_embs=None,
        down_patch_size=7,
        down_stride=2,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        cls_ratio=2.0,
        inference_mode=False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        if pos_embs is None:
            pos_embs = [None] * len(layers)
        if se_downsamples is None:
            se_downsamples = [False] * len(layers)

        self.patch_embed = convolutional_stem(3, embed_dims[0], inference_mode)

        network = []
        for i in range(len(layers)):
            if pos_embs[i] is not None:
                network.append(
                    pos_embs[i](
                        embed_dims[i], embed_dims[i], inference_mode=inference_mode
                    )
                )
            
            # Simple stage builder
            stage = nn.LayerList()
            for _ in range(layers[i]):
                if token_mixers[i] == "repmixer":
                    stage.append(RepMixerBlock(
                        embed_dims[i], kernel_size=repmixer_kernel_size,
                        mlp_ratio=mlp_ratios[i], act_layer=act_layer,
                        inference_mode=inference_mode
                    ))
                else:
                    stage.append(AttentionBlock(
                        embed_dims[i], mlp_ratio=mlp_ratios[i],
                        act_layer=act_layer, norm_layer=norm_layer
                    ))
            network.append(stage)

            if i < len(layers) - 1:
                if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                    network.append(PatchEmbed(
                        patch_size=down_patch_size, stride=down_stride,
                        in_channels=embed_dims[i], embed_dim=embed_dims[i + 1],
                        inference_mode=inference_mode, use_se=se_downsamples[i + 1]
                    ))
        
        self.network = nn.LayerList(network)
        self.conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3, stride=1, padding=1, groups=embed_dims[-1],
            inference_mode=inference_mode, use_se=True, num_conv_branches=1
        )
        self.head = nn.Linear(int(embed_dims[-1] * cls_ratio), num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.patch_embed(x)
        for block in self.network:
            if isinstance(block, nn.LayerList):
                for sub_block in block:
                    x = sub_block(x)
            else:
                x = block(x)
        x = self.conv_exp(x)
        # 不要在这里池化，让外层的 GlobalPool2D 层处理
        x = self.head(x)
        return x


def mci0(pretrained=False, **kwargs):
    layers = [2, 6, 10, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    return FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios, downsamples=downsamples,
        se_downsamples=se_downsamples, **kwargs
    )


def mci1(pretrained=False, **kwargs):
    layers = [4, 12, 20, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    return FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios, downsamples=downsamples,
        se_downsamples=se_downsamples, **kwargs
    )


def mci2(pretrained=False, **kwargs):
    layers = [4, 12, 24, 4]
    embed_dims = [80, 160, 320, 640]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    se_downsamples = [False, False, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")
    return FastViT(
        layers, token_mixers=token_mixers, embed_dims=embed_dims,
        pos_embs=pos_embs, mlp_ratios=mlp_ratios, downsamples=downsamples,
        se_downsamples=se_downsamples, **kwargs
    )
