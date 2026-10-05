from functools import partial
from typing import Optional, Tuple, Type, Union

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    # EfficientViT-B0 (only 224 resolution available)
    "efficientvit_b0":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b0.r224_in1k.pdparams",
    # EfficientViT-B1
    "efficientvit_b1":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b1.r224_in1k.pdparams",
    "efficientvit_b1_r256":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b1.r256_in1k.pdparams",
    "efficientvit_b1_r288":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b1.r288_in1k.pdparams",
    # EfficientViT-B2
    "efficientvit_b2":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b2.r224_in1k.pdparams",
    "efficientvit_b2_r256":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b2.r256_in1k.pdparams",
    "efficientvit_b2_r288":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b2.r288_in1k.pdparams",
    # EfficientViT-B3
    "efficientvit_b3":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b3.r224_in1k.pdparams",
    "efficientvit_b3_r256":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b3.r256_in1k.pdparams",
    "efficientvit_b3_r288":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_b3.r288_in1k.pdparams",
    # EfficientViT-L1 (only 224 resolution available)
    "efficientvit_l1":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l1.r224_in1k.pdparams",
    # EfficientViT-L2
    "efficientvit_l2":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l2.r224_in1k.pdparams",
    "efficientvit_l2_r256":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l2.r256_in1k.pdparams",
    "efficientvit_l2_r288":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l2.r288_in1k.pdparams",
    "efficientvit_l2_r384":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l2.r384_in1k.pdparams",
    # EfficientViT-L3
    "efficientvit_l3":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l3.r224_in1k.pdparams",
    "efficientvit_l3_r256":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l3.r256_in1k.pdparams",
    "efficientvit_l3_r320":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l3.r320_in1k.pdparams",
    "efficientvit_l3_r384":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/efficientvit_l3.r384_in1k.pdparams",
}

__all__ = [
    "EfficientVit",
    "EfficientVitLarge",
    "Efficientvit_B0",
    "Efficientvit_B1",
    "Efficientvit_B2",
    "Efficientvit_B3",
    "Efficientvit_L1",
    "Efficientvit_L2",
    "Efficientvit_L3",
]


def val2list(x: Union[list, tuple, any], repeat_time: int = 1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(
    x: Union[list, tuple, any], min_len: int = 1, idx_repeat: int = -1
) -> tuple:
    x = val2list(x)
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]
    return tuple(x)


def get_same_padding(
    kernel_size: Union[int, Tuple[int, ...]]
) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


class ConvNormAct(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        dropout: float = 0.0,
        norm_layer: Optional[Type[nn.Layer]] = nn.BatchNorm2D,
        act_layer: Optional[Type[nn.Layer]] = nn.ReLU,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        padding = get_same_padding(kernel_size)
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )
        self.norm = norm_layer(out_channels) if norm_layer else nn.Identity()
        self.act = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class DSConv(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        use_bias: Union[bool, Tuple[bool, bool]] = False,
        norm_layer: Union[
            Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]
        ] = nn.BatchNorm2D,
        act_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            nn.ReLU6,
            None,
        ),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        self.depth_conv = ConvNormAct(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            in_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class ConvBlock(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 1,
        use_bias: Union[bool, Tuple[bool, bool]] = False,
        norm_layer: Union[
            Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]
        ] = nn.BatchNorm2D,
        act_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            nn.ReLU6,
            None,
        ),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.conv1 = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.conv2 = ConvNormAct(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class MBConv(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        use_bias: Union[bool, Tuple[bool, ...]] = False,
        norm_layer: Union[
            Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]
        ] = nn.BatchNorm2D,
        act_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            nn.ReLU6,
            nn.ReLU6,
            None,
        ),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm_layer = val2tuple(norm_layer, 3)
        act_layer = val2tuple(act_layer, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.inverted_conv = ConvNormAct(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.depth_conv = ConvNormAct(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        mid_channels: Optional[int] = None,
        expand_ratio: float = 6,
        groups: int = 1,
        use_bias: Union[bool, Tuple[bool, ...]] = False,
        norm_layer: Union[
            Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]
        ] = nn.BatchNorm2D,
        act_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            nn.ReLU6,
            None,
        ),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        mid_channels = mid_channels or round(in_channels * expand_ratio)
        self.spatial_conv = ConvNormAct(
            in_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=groups,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            bias=use_bias[0],
        )
        self.point_conv = ConvNormAct(
            mid_channels,
            out_channels,
            1,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            bias=use_bias[1],
        )

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResidualBlock(nn.Layer):
    def __init__(
        self,
        main: Optional[nn.Layer],
        shortcut: Optional[nn.Layer] = None,
        pre_norm: Optional[nn.Layer] = None,
    ):
        super().__init__()
        self.pre_norm = pre_norm if pre_norm is not None else nn.Identity()
        self.main = main
        self.shortcut = shortcut

    def forward(self, x):
        res = self.main(self.pre_norm(x))
        if self.shortcut is not None:
            res = res + self.shortcut(x)
        return res


def build_local_block(
    in_channels: int,
    out_channels: int,
    stride: int,
    expand_ratio: float,
    norm_layer: Type[nn.Layer],
    act_layer: Type[nn.Layer],
    fewer_norm: bool = False,
    block_type: str = "default",
):
    assert block_type in ["default", "large", "fused"]
    if expand_ratio == 1:
        if block_type == "default":
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
        else:
            block = ConvBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    else:
        if block_type == "default":
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm_layer=(None, None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, act_layer, None),
            )
        else:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm_layer=(None, norm_layer) if fewer_norm else norm_layer,
                act_layer=(act_layer, None),
            )
    return block


class LiteMLA(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: Optional[int] = None,
        heads_ratio: float = 1.0,
        dim: int = 8,
        use_bias: Union[bool, Tuple[bool, ...]] = False,
        norm_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            None,
            nn.BatchNorm2D,
        ),
        act_layer: Union[Type[nn.Layer], Tuple[Optional[Type[nn.Layer]], ...]] = (
            None,
            None,
        ),
        kernel_func: Type[nn.Layer] = nn.ReLU,
        scales: Tuple[int, ...] = (5,),
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = paddle.to_tensor(eps, dtype="float32")
        heads = heads or int(in_channels // dim * heads_ratio)
        total_dim = heads * dim
        use_bias = val2tuple(use_bias, 2)
        norm_layer = val2tuple(norm_layer, 2)
        act_layer = val2tuple(act_layer, 2)
        self.dim = dim
        self.heads = heads
        self.scales = scales
        self.total_dim = total_dim
        self.num_scales = len(scales)
        self.qkv = ConvNormAct(
            in_channels,
            3 * total_dim,
            1,
            bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.aggreg = nn.LayerList()
        for scale in scales:
            self.aggreg.append(
                nn.Sequential(
                    nn.Conv2D(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias_attr=use_bias[0],
                    ),
                    nn.Conv2D(
                        3 * total_dim,
                        3 * total_dim,
                        1,
                        groups=3 * heads,
                        bias_attr=use_bias[0],
                    ),
                )
            )
        self.kernel_func = kernel_func()
        self.proj = ConvNormAct(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def _attn(self, q, k, v):
        dtype = v.dtype
        q, k, v = q.astype("float32"), k.astype("float32"), v.astype("float32")
        kt = k.transpose([0, 1, 3, 2])
        kv = kt @ v
        out = q @ kv
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        return out.astype(dtype)

    def forward(self, x):
        B, C_in, H, W = x.shape
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = paddle.concat(multi_scale_qkv, axis=1)
        num_heads = self.heads * (1 + self.num_scales)
        HW = H * W
        multi_scale_qkv = paddle.reshape(
            multi_scale_qkv, [B, num_heads, 3 * self.dim, HW]
        )
        multi_scale_qkv = paddle.transpose(multi_scale_qkv, [0, 1, 3, 2])
        q, k, v = paddle.split(multi_scale_qkv, 3, axis=-1)
        q = self.kernel_func(q)
        k = self.kernel_func(k)
        v_pad = paddle.ones([B, num_heads, HW, 1], dtype=v.dtype)
        v = paddle.concat([v, v_pad], axis=-1)
        out = self._attn(q, k, v)
        C_out = self.total_dim * (1 + self.num_scales)
        out = paddle.transpose(out, [0, 1, 3, 2])
        out = paddle.reshape(out, [B, C_out, H, W])
        out = self.proj(out)
        return out


class EfficientVitBlock(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        head_dim: int = 32,
        expand_ratio: float = 4,
        norm_layer: Type[nn.Layer] = nn.BatchNorm2D,
        act_layer: Type[nn.Layer] = nn.Hardswish,
    ):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=head_dim,
                norm_layer=(None, norm_layer),
            ),
            nn.Identity(),
        )
        self.local_module = ResidualBlock(
            MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            ),
            nn.Identity(),
        )

    def forward(self, x):
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class Stem(nn.Sequential):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int,
        norm_layer: Type[nn.Layer],
        act_layer: Type[nn.Layer],
        block_type: str = "default",
    ):
        super().__init__()
        self.stride = 2
        self.add_sublayer(
            "in_conv",
            ConvNormAct(
                in_chs,
                out_chs,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                act_layer=act_layer,
            ),
        )
        stem_block = 0
        for _ in range(depth):
            self.add_sublayer(
                f"res{stem_block}",
                ResidualBlock(
                    build_local_block(
                        in_channels=out_chs,
                        out_channels=out_chs,
                        stride=1,
                        expand_ratio=1,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                        block_type=block_type,
                    ),
                    nn.Identity(),
                ),
            )
            stem_block += 1


class EfficientVitStage(nn.Layer):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int,
        norm_layer: Type[nn.Layer],
        act_layer: Type[nn.Layer],
        expand_ratio: float,
        head_dim: int,
        vit_stage: bool = False,
    ):
        super().__init__()
        blocks = [
            ResidualBlock(
                build_local_block(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    stride=2,
                    expand_ratio=expand_ratio,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    fewer_norm=vit_stage,
                ),
                None,
            )
        ]
        in_chs = out_chs
        if vit_stage:
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=expand_ratio,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            for i in range(1, depth):
                blocks.append(
                    ResidualBlock(
                        build_local_block(
                            in_channels=in_chs,
                            out_channels=out_chs,
                            stride=1,
                            expand_ratio=expand_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                        ),
                        nn.Identity(),
                    )
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class EfficientVitLargeStage(nn.Layer):
    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        depth: int,
        norm_layer: Type[nn.Layer],
        act_layer: Type[nn.Layer],
        head_dim: int,
        vit_stage: bool = False,
        fewer_norm: bool = False,
    ):
        super().__init__()
        blocks = [
            ResidualBlock(
                build_local_block(
                    in_channels=in_chs,
                    out_channels=out_chs,
                    stride=2,
                    expand_ratio=24 if vit_stage else 16,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    fewer_norm=vit_stage or fewer_norm,
                    block_type="default" if fewer_norm else "fused",
                ),
                None,
            )
        ]
        in_chs = out_chs
        if vit_stage:
            for _ in range(depth):
                blocks.append(
                    EfficientVitBlock(
                        in_channels=in_chs,
                        head_dim=head_dim,
                        expand_ratio=6,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                )
        else:
            for i in range(depth):
                blocks.append(
                    ResidualBlock(
                        build_local_block(
                            in_channels=in_chs,
                            out_channels=out_chs,
                            stride=1,
                            expand_ratio=4,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            fewer_norm=fewer_norm,
                            block_type="default" if fewer_norm else "fused",
                        ),
                        nn.Identity(),
                    )
                )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)


class ClassifierHead(nn.Layer):
    def __init__(
        self,
        in_channels: int,
        widths: Tuple[int, int],
        num_classes: int = 1000,
        dropout: float = 0.0,
        norm_layer: type = nn.BatchNorm2D,
        act_layer: type = nn.Hardswish,
        pool_type: str = "avg",
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        self.widths = widths
        self.num_features = widths[-1]
        self.in_conv = ConvNormAct(
            in_channels, widths[0], 1, norm_layer=norm_layer, act_layer=act_layer
        )
        self.global_pool = nn.AdaptiveAvgPool2D(output_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(widths[0], widths[1], bias_attr=False),
            nn.LayerNorm(widths[1], epsilon=norm_eps),
            act_layer() if act_layer is not None else nn.Identity(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(widths[1], num_classes, bias_attr=True)
            if num_classes > 0
            else nn.Identity(),
        )

    def forward(self, x, pre_logits: bool = False):
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = paddle.flatten(x, 1)
        if pre_logits:
            for i in range(len(self.classifier) - 1):
                x = self.classifier[i](x)
        else:
            x = self.classifier(x)
        return x


class EfficientVit(nn.Layer):
    def __init__(
        self,
        in_chans: int = 3,
        widths: Tuple[int, ...] = (),
        depths: Tuple[int, ...] = (),
        head_dim: int = 32,
        expand_ratio: float = 4,
        norm_layer: type = nn.BatchNorm2D,
        act_layer: type = nn.Hardswish,
        global_pool: str = "avg",
        head_widths: Tuple[int, ...] = (),
        drop_rate: float = 0.0,
        num_classes: int = 1000,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.stem = Stem(in_chans, widths[0], depths[0], norm_layer, act_layer)
        stages = []
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            stages.append(
                EfficientVitStage(
                    in_channels,
                    w,
                    depth=d,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    expand_ratio=expand_ratio,
                    head_dim=head_dim,
                    vit_stage=i >= 2,
                )
            )
            in_channels = w
        self.stages = nn.Sequential(*stages)
        self.num_features = in_channels
        self.head = ClassifierHead(
            self.num_features,
            widths=head_widths,
            num_classes=num_classes,
            dropout=drop_rate,
            pool_type=self.global_pool,
        )
        self.head_hidden_size = self.head.num_features

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class EfficientVitLarge(nn.Layer):
    def __init__(
        self,
        in_chans: int = 3,
        widths: Tuple[int, ...] = (),
        depths: Tuple[int, ...] = (),
        head_dim: int = 32,
        norm_layer: type = nn.BatchNorm2D,
        act_layer: type = None,
        global_pool: str = "avg",
        head_widths: Tuple[int, ...] = (),
        drop_rate: float = 0.0,
        num_classes: int = 1000,
        norm_eps: float = 1e-7,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.norm_eps = norm_eps
        if act_layer is None:

            class CorrectGELU(nn.Layer):
                def forward(self, x):
                    return paddle.nn.functional.gelu(x, approximate=True)

            act_layer = CorrectGELU
        norm_layer = partial(norm_layer, epsilon=self.norm_eps)
        self.stem = Stem(
            in_chans, widths[0], depths[0], norm_layer, act_layer, block_type="large"
        )
        stages = []
        in_channels = widths[0]
        for i, (w, d) in enumerate(zip(widths[1:], depths[1:])):
            stages.append(
                EfficientVitLargeStage(
                    in_channels,
                    w,
                    depth=d,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    head_dim=head_dim,
                    vit_stage=i >= 3,
                    fewer_norm=i >= 2,
                )
            )
            in_channels = w
        self.stages = nn.Sequential(*stages)
        self.num_features = in_channels
        self.head = ClassifierHead(
            self.num_features,
            widths=head_widths,
            num_classes=num_classes,
            dropout=drop_rate,
            pool_type=self.global_pool,
            act_layer=act_layer,
            norm_eps=self.norm_eps,
        )
        self.head_hidden_size = self.head.num_features

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


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


def Efficientvit_B0(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(8, 16, 32, 64, 128),
        depths=(1, 2, 2, 2, 2),
        head_dim=16,
        head_widths=(1024, 1280),
    )
    model_args.update(kwargs)
    model = EfficientVit(**model_args)
    # Only 224 resolution available for B0
    model_url = MODEL_URLS["efficientvit_b0"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_B1(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(16, 32, 64, 128, 256),
        depths=(1, 2, 3, 3, 4),
        head_dim=16,
        head_widths=(1536, 1600),
    )
    model_args.update(kwargs)
    model = EfficientVit(**model_args)
    # Select model URL based on image size
    if img_size == 256:
        model_url = MODEL_URLS["efficientvit_b1_r256"]
    elif img_size == 288:
        model_url = MODEL_URLS["efficientvit_b1_r288"]
    else:
        model_url = MODEL_URLS["efficientvit_b1"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_B2(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(24, 48, 96, 192, 384),
        depths=(1, 3, 4, 4, 6),
        head_dim=32,
        head_widths=(2304, 2560),
    )
    model_args.update(kwargs)
    model = EfficientVit(**model_args)
    # Select model URL based on image size
    if img_size == 256:
        model_url = MODEL_URLS["efficientvit_b2_r256"]
    elif img_size == 288:
        model_url = MODEL_URLS["efficientvit_b2_r288"]
    else:
        model_url = MODEL_URLS["efficientvit_b2"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_B3(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(32, 64, 128, 256, 512),
        depths=(1, 4, 6, 6, 9),
        head_dim=32,
        head_widths=(2304, 2560),
    )
    model_args.update(kwargs)
    model = EfficientVit(**model_args)
    # Select model URL based on image size
    if img_size == 256:
        model_url = MODEL_URLS["efficientvit_b3_r256"]
    elif img_size == 288:
        model_url = MODEL_URLS["efficientvit_b3_r288"]
    else:
        model_url = MODEL_URLS["efficientvit_b3"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_L1(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(32, 64, 128, 256, 512),
        depths=(1, 1, 1, 6, 6),
        head_dim=32,
        head_widths=(3072, 3200),
    )
    model_args.update(kwargs)
    model = EfficientVitLarge(**model_args)
    # Only 224 resolution available for L1
    model_url = MODEL_URLS["efficientvit_l1"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_L2(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(32, 64, 128, 256, 512),
        depths=(1, 2, 2, 8, 8),
        head_dim=32,
        head_widths=(3072, 3200),
    )
    model_args.update(kwargs)
    model = EfficientVitLarge(**model_args)
    # Select model URL based on image size
    if img_size == 256:
        model_url = MODEL_URLS["efficientvit_l2_r256"]
    elif img_size == 288:
        model_url = MODEL_URLS["efficientvit_l2_r288"]
    elif img_size == 384:
        model_url = MODEL_URLS["efficientvit_l2_r384"]
    else:
        model_url = MODEL_URLS["efficientvit_l2"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model


def Efficientvit_L3(pretrained=False, use_ssld=False, img_size=224, **kwargs):
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model_args = dict(
        widths=(64, 128, 256, 512, 1024),
        depths=(1, 2, 2, 8, 8),
        head_dim=32,
        head_widths=(6144, 6400),
    )
    model_args.update(kwargs)
    model = EfficientVitLarge(**model_args)
    # Select model URL based on image size
    if img_size == 256:
        model_url = MODEL_URLS["efficientvit_l3_r256"]
    elif img_size == 320:
        model_url = MODEL_URLS["efficientvit_l3_r320"]
    elif img_size == 384:
        model_url = MODEL_URLS["efficientvit_l3_r384"]
    else:
        model_url = MODEL_URLS["efficientvit_l3"]
    _load_pretrained(
        pretrained, model, model_url, use_ssld=use_ssld
    )
    return model
