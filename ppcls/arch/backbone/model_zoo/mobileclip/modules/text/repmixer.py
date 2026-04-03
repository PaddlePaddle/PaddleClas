# -*- coding: utf-8 -*-
from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from mobileclip.modules.common.mobileone import MobileOneBlock


class ConvFFN(nn.Layer):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        context_size: int,
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
                kernel_size=(1, int(context_size)),
                padding=(0, int(context_size // 2)),
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
                kernel_size=(1, self.kernel_size),
                stride=1,
                padding=(0, self.kernel_size // 2),
                groups=self.dim,
                bias_attr=True,
            )
        else:
            self.norm = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
                groups=dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
            )
            self.mixer = MobileOneBlock(
                dim,
                dim,
                (1, kernel_size),
                padding=(0, kernel_size // 2),
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


class RepMixerBlock(nn.Layer):
    def __init__(
        self,
        dim: int,
        kernel_size: int = 11,
        mlp_ratio: float = 4.0,
        act_layer: nn.Layer = nn.GELU,
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        *args,
        **kwargs,
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
            context_size=kernel_size,
            hidden_channels=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = nn.Identity() # Placeholder

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        # RepMixer expects 4D input [B, C, 1, L]
        x = x.transpose([0, 2, 1]).unsqueeze(2)
        x = x + self.drop_path(self.token_mixer(x))
        x = x + self.drop_path(self.convffn(x))
        x = x.squeeze(2).transpose([0, 2, 1])
        return x
