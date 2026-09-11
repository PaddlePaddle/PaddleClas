"""CPUBone backbone implemented in Paddle.

Migrated from timm 1.0.29 (timm/models/cpubone.py), original implementation:
https://github.com/altair199797/CPUBone.

State dict keys match timm native checkpoints. timm GroupNorm1 is expressed
as GroupNorm(num_groups=1); fused SDPA is replaced by an equivalent manual
attention; norm/act layers are fixed to BatchNorm2D/Hardswish as in every
released variant. Top-level class inherits TheseusLayer so the PaddleClas
pruning/quantization pipeline applies.
"""

from typing import Dict, List, Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..base.theseus_layer import TheseusLayer

__all__ = [
    'CPUBone_nano', 'CPUBone_t0', 'CPUBone_s0',
    'CPUBone_b0_bfrobust', 'CPUBone_b1_bfrobust', 'CPUBone_b1_dwnorm',
    'CPUBone_b2_bfrobust', 'CPUBone_b2pt5_dwnorm', 'CPUBone_b3',
]

# Official BOS hosting is not available yet; pass a local converted
# .pdparams path via pretrained=<path> instead of pretrained=True.
MODEL_URLS = {
    "CPUBone_nano": "",
    "CPUBone_t0": "",
    "CPUBone_s0": "",
    "CPUBone_b0_bfrobust": "",
    "CPUBone_b1_bfrobust": "",
    "CPUBone_b1_dwnorm": "",
    "CPUBone_b2_bfrobust": "",
    "CPUBone_b2pt5_dwnorm": "",
    "CPUBone_b3": "",
}

_LOCAL_MBCONV_NORM_MODES = {
    # mode: (expand, depthwise, project)
    'proj': (False, False, True),
    'depth_proj': (False, True, True),
    'all': (True, True, True),
}


def _check_local_mbconv_norm(local_mbconv_norm):
    if local_mbconv_norm not in _LOCAL_MBCONV_NORM_MODES:
        raise ValueError(
            'Invalid local_mbconv_norm={!r}; expected one of {}.'.format(
                local_mbconv_norm, tuple(_LOCAL_MBCONV_NORM_MODES)))


def _check_global_pool(global_pool):
    assert global_pool in ("", "avg"), "CPUBone only supports average or disabled pooling"


def get_same_padding(kernel_size, stride=1):
    # kernel_size 2 at stride 1 needs an asymmetric left/top pad of 1,
    # signalled by -1 and handled in ConvLayer with ZeroPad2D.
    if kernel_size == 2:
        return 0 if stride == 2 else -1
    assert kernel_size % 2 > 0, "kernel size should be odd number"
    return kernel_size // 2


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    return (x / keep_prob) * random_tensor


class DropPath(nn.Layer):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def remap_legacy_state_dict(state_dict):
    """Remap keys of original-repo (non-timm) checkpoints to this layout."""
    remapped = {}
    for k, v in state_dict.items():
        k = k.replace(".conv_proj.0.", ".conv_proj.conv.")
        k = k.replace(".conv_proj.1.", ".conv_proj.norm.")
        k = k.replace(".pwise.0.", ".pwise.")
        k = k.replace("head.op_list.0.", "head.in_conv.")
        k = k.replace("head.op_list.2.", "head.pre_classifier.")
        k = k.replace("head.op_list.3.", "head.classifier.")
        k = k.replace("backbone.input_stem.", "stem.")
        k = k.replace("backbone.stages.", "stages.")
        k = k.replace(".op_list.", ".")
        remapped[k] = v
    return remapped


class LinearLayer(nn.Layer):
    def __init__(self,
                 in_features,
                 out_features,
                 use_bias=True,
                 dropout=0.,
                 norm_layer=None,
                 act_layer=None):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, bias_attr=use_bias)
        self.norm = norm_layer(out_features) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ResidualBlock(nn.Layer):
    def __init__(self, main, shortcut=None, drop_path=0.):
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        if self.shortcut is None:
            return self.main(x)
        return self.drop_path(self.main(x)) + self.shortcut(x)


class ConvLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 use_bias=False,
                 norm_layer=nn.BatchNorm2D,
                 act_layer=nn.ReLU):
        super().__init__()
        padding = get_same_padding(kernel_size, stride)
        if padding == -1:
            self.conv = nn.Sequential(
                nn.ZeroPad2D((1, 0, 1, 0)),
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=0,
                    groups=groups,
                    bias_attr=use_bias,
                ),
            )
        else:
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=use_bias,
            )
        self.norm = norm_layer(out_channels) if norm_layer is not None else None
        self.act = act_layer() if act_layer is not None else None

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class MBConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 mid_channels=None,
                 expand_ratio=6,
                 expand_groups=1,
                 use_bias=(False, False, False),
                 norm_layer=(nn.BatchNorm2D, nn.BatchNorm2D, nn.BatchNorm2D),
                 act_layer=(nn.ReLU6, nn.ReLU6, None)):
        super().__init__()
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            groups=expand_groups,
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            norm_layer=norm_layer[2],
            act_layer=act_layer[2],
            use_bias=use_bias[2],
        )

    def forward(self, x):
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 mid_channels=None,
                 expand_ratio=6,
                 expand_groups=1,
                 use_bias=(False, False),
                 norm_layer=(nn.BatchNorm2D, nn.BatchNorm2D),
                 act_layer=(nn.ReLU6, None)):
        super().__init__()
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=expand_groups,
            use_bias=use_bias[0],
            norm_layer=norm_layer[0],
            act_layer=act_layer[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            groups=1,
            use_bias=use_bias[1],
            norm_layer=norm_layer[1],
            act_layer=act_layer[1],
        )

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ConvAttention(nn.Layer):
    """Conv-style attention: depthwise downsample, 1x1 qkv projection,
    multi-head softmax attention, then upsample back and crop.

    Manual attention replaces fused SDPA (numerically equal in fp32).
    Paddle F.pad / ZeroPad2D tuple order matches torch
    (left, right, top, bottom), verified by tests.
    """

    def __init__(self,
                 input_dim,
                 head_dim_mul=1.0,
                 att_stride=4,
                 att_kernel=7,
                 fuse_out_proj=False,
                 small_kernels=False,
                 upsample_mode='transpose'):
        super().__init__()
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.num_keys = 3
        self.scale = self.head_dim ** -0.5
        self.att_stride = att_stride
        self.small_kernels = small_kernels

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvLayer(
            input_dim,
            input_dim,
            kernel_size=2 if small_kernels else att_kernel,
            stride=att_stride,
            groups=input_dim,
            norm_layer=nn.BatchNorm2D,
            act_layer=None,
        )
        self.pwise = nn.Conv2D(input_dim, total_dim, kernel_size=1, stride=1, padding=0,
                               bias_attr=False)

        self.o_proj_inpdim = self.head_dim * self.num_heads
        if fuse_out_proj:
            self.o_proj = nn.Identity()
        else:
            self.o_proj = nn.Conv2D(self.o_proj_inpdim, input_dim, kernel_size=1,
                                    stride=1, padding=0)

        if upsample_mode == 'nearest':
            upsampling = [nn.Upsample(scale_factor=att_stride, mode='nearest')
                          if att_stride > 1 else nn.Identity()]
            if fuse_out_proj:
                upsampling = [nn.Conv2D(self.o_proj_inpdim, input_dim, kernel_size=1,
                                        stride=1, padding=0)] + upsampling
            self.upsampling = nn.Sequential(*upsampling)
        elif fuse_out_proj:
            if att_stride == 1:
                self.upsampling = nn.Conv2DTranspose(self.o_proj_inpdim, input_dim,
                                                     kernel_size=3, stride=1, padding=1)
            else:
                self.upsampling = nn.Conv2DTranspose(
                    self.o_proj_inpdim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride,
                    padding=att_stride // 2)
        else:
            if att_stride == 1:
                self.upsampling = nn.Conv2DTranspose(
                    input_dim, input_dim, kernel_size=3, stride=1, padding=1,
                    groups=input_dim)
            else:
                self.upsampling = nn.Conv2DTranspose(
                    input_dim, input_dim,
                    kernel_size=att_stride * 2, stride=att_stride,
                    padding=att_stride // 2, groups=input_dim)

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        if self.small_kernels and self.att_stride > 1:
            pad_h = (-H) % self.att_stride
            pad_w = (-W) % self.att_stride
            if pad_h or pad_w:
                x = F.pad(x, (0, pad_w, 0, pad_h))

        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N = xout.shape[0]
        h = xout.shape[2]
        w = xout.shape[3]
        qkv = xout.reshape([N, self.num_heads, self.num_keys * self.head_dim, h * w])
        qkv = qkv.transpose([0, 1, 3, 2])
        q, k, v = qkv.chunk(3, axis=3)

        q = q * self.scale
        attn = q @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        values = attn @ v
        o = self.o_proj(values.transpose([0, 1, 3, 2]).reshape([N, self.o_proj_inpdim, h, w]))

        o = self.upsampling(o)
        return o[:, :, :H, :W]


class CPUBoneBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 expand_ratio=4,
                 act_layer=nn.Hardswish,
                 fused_conv=False,
                 expand_groups=1,
                 att_stride=1,
                 mlp_ratio=4,
                 small_kernels=False,
                 attn_upsample='transpose',
                 proj_drop=0.1,
                 drop_path=0.,
                 local_mbconv_norm='proj'):
        super().__init__()
        _check_local_mbconv_norm(local_mbconv_norm)
        att_kernel = 5 if att_stride > 1 else 3
        norm_layer = nn.BatchNorm2D

        block = ConvAttention(
            input_dim=in_channels,
            att_stride=att_stride,
            att_kernel=att_kernel,
            head_dim_mul=0.5,
            fuse_out_proj=fused_conv,
            small_kernels=small_kernels,
            upsample_mode=attn_upsample,
        )

        context_module = ResidualBlock(
            nn.Sequential(nn.GroupNorm(num_groups=1, num_channels=in_channels), block),
            nn.Identity(), drop_path)
        mlp = nn.Sequential(
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.Conv2D(in_channels, in_channels * mlp_ratio, kernel_size=1),
            nn.GELU(),
            nn.Conv2D(in_channels * mlp_ratio, in_channels, kernel_size=1),
            nn.Dropout(p=proj_drop),
        )
        context_module = nn.Sequential(context_module,
                                       ResidualBlock(mlp, nn.Identity(), drop_path))

        if fused_conv and in_channels < 256:
            local_module = FusedMBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                use_bias=(True, False),
                kernel_size=2 if small_kernels else 3,
                expand_groups=expand_groups,
                norm_layer=(norm_layer, norm_layer),
                act_layer=(act_layer, None),
            )
        else:
            norm_mask = _LOCAL_MBCONV_NORM_MODES[local_mbconv_norm]
            local_norms = tuple(norm_layer if enabled else None for enabled in norm_mask)
            local_biases = tuple(not enabled for enabled in norm_mask)
            local_module = MBConv(
                in_channels=in_channels,
                out_channels=in_channels,
                expand_ratio=expand_ratio,
                expand_groups=expand_groups,
                use_bias=local_biases,
                kernel_size=2 if small_kernels else 3,
                norm_layer=local_norms,
                act_layer=(act_layer, act_layer, None),
            )

        self.total = nn.Sequential(context_module,
                                   ResidualBlock(local_module, nn.Identity(), drop_path))

    def forward(self, x):
        return self.total(x)


class ClsHead(nn.Layer):
    def __init__(self,
                 in_channels,
                 width_list,
                 num_classes=1000,
                 global_pool="avg",
                 dropout=0.0,
                 norm_layer=nn.BatchNorm2D,
                 act_layer=nn.Hardswish):
        super().__init__()
        _check_global_pool(global_pool)
        self.num_features = width_list[-1]
        self.dropout = dropout
        self.pool_type = global_pool

        self.in_conv = ConvLayer(in_channels, width_list[0], 1, norm_layer=norm_layer,
                                 act_layer=act_layer)
        self.global_pool = nn.AdaptiveAvgPool2D(output_size=1) if global_pool else nn.Identity()
        self.flatten = nn.Flatten(start_axis=1) if global_pool else nn.Identity()
        self.pre_classifier = LinearLayer(
            width_list[0], width_list[1], False, norm_layer=nn.LayerNorm,
            act_layer=act_layer)
        self.classifier = (
            LinearLayer(width_list[1], num_classes, True, dropout) if num_classes > 0
            else nn.Identity())

    def reset(self, num_classes, global_pool=None):
        """Reset the classifier head, mirroring timm's reset interface."""
        if global_pool is not None:
            _check_global_pool(global_pool)
            self.pool_type = global_pool
            self.global_pool = nn.AdaptiveAvgPool2D(output_size=1) if global_pool \
                else nn.Identity()
            self.flatten = nn.Flatten(start_axis=1) if global_pool else nn.Identity()
        if num_classes > 0:
            self.classifier = LinearLayer(self.num_features, num_classes, True, self.dropout)
        else:
            self.classifier = nn.Identity()

    def forward(self, x, pre_logits=False):
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if not self.pool_type:
            x = x.transpose([0, 2, 3, 1])
        x = self.pre_classifier(x)
        if not pre_logits:
            x = self.classifier(x)
        if not self.pool_type:
            x = x.transpose([0, 3, 1, 2])
        return x


class CPUBone(TheseusLayer):
    def __init__(self,
                 width_list,
                 depth_list,
                 in_chans=3,
                 num_classes=1000,
                 global_pool="avg",
                 head_widths=(1536, 1600),
                 drop_rate=0.0,
                 proj_drop_rate=0.1,
                 drop_path_rate=0.0,
                 expand_ratio=4,
                 fused_conv=False,
                 fused_downsample=False,
                 attn_mlp_ratio=2,
                 stem_expand_ratio=2,
                 downsample_expand_ratios=None,
                 expand_groups=1,
                 small_kernels=False,
                 attn_upsample='transpose',
                 local_mbconv_norm='proj'):
        super().__init__()
        _check_global_pool(global_pool)
        assert attn_upsample in ('transpose', 'nearest')
        _check_local_mbconv_norm(local_mbconv_norm)
        num_stages = len(width_list) - 1
        if downsample_expand_ratios is None:
            downsample_expand_ratios = (expand_ratio,) * num_stages
        assert len(downsample_expand_ratios) == num_stages
        self.num_classes = num_classes
        self.num_features = width_list[-1]
        self.head_hidden_size = head_widths[-1]
        self.global_pool = global_pool

        self.expand_ratio = expand_ratio
        self.act_layer = nn.Hardswish
        self.fused_conv = fused_conv
        self.fused_downsample = fused_downsample
        self.attn_mlp_ratio = attn_mlp_ratio
        self.proj_drop_rate = proj_drop_rate
        self.stem_expand_ratio = stem_expand_ratio
        self.downsample_expand_ratios = tuple(downsample_expand_ratios)
        self.expand_groups = expand_groups
        self.small_kernels = small_kernels
        self.attn_upsample = attn_upsample
        self.local_mbconv_norm = local_mbconv_norm

        dpr = self._calculate_drop_path_rates(drop_path_rate, sum(depth_list))

        self.stem, in_channels = self._build_stem(in_chans, width_list[0], depth_list[0],
                                                  dpr[:depth_list[0]])
        block_idx = depth_list[0]

        stages = []
        for stage_num, (width, depth) in enumerate(zip(width_list[1:], depth_list[1:]),
                                                   start=1):
            stage_dpr = dpr[block_idx:block_idx + depth]
            block_idx += depth
            if stage_num >= 3:
                blocks, in_channels = self._build_attention_stage(in_channels, width, depth,
                                                                  stage_num, stage_dpr)
            else:
                blocks, in_channels = self._build_conv_stage(in_channels, width, depth,
                                                             stage_num, stage_dpr)
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        self.head = ClsHead(
            in_channels=width_list[-1],
            width_list=list(head_widths),
            num_classes=num_classes,
            global_pool=global_pool,
            dropout=drop_rate,
            norm_layer=nn.BatchNorm2D,
            act_layer=self.act_layer,
        )

    @staticmethod
    def _calculate_drop_path_rates(drop_path_rate, total):
        # per-block linear ramp, same as timm calculate_drop_path_rates
        if total <= 1:
            return [0.0] * max(total, 0)
        return [drop_path_rate * i / (total - 1) for i in range(total)]

    def _build_stem(self, in_channels, stem_width, depth, dpr):
        blocks = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=stem_width,
                kernel_size=3,
                stride=2,
                norm_layer=nn.BatchNorm2D,
                act_layer=self.act_layer,
            )
        ]
        in_channels = stem_width
        for i in range(depth):
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=self.stem_expand_ratio,
                fusedmbconv=self.fused_conv,
                expand_groups=self.expand_groups,
                norm_layer=nn.BatchNorm2D,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity(), dpr[i]))
        return nn.Sequential(*blocks), in_channels

    def _build_conv_stage(self, in_channels, width, depth, stage_num, dpr):
        blocks = []
        for i in range(depth):
            stride = 2 if i == 0 else 1
            block = self.build_local_block(
                in_channels=in_channels,
                out_channels=width,
                stride=stride,
                expand_ratio=self.downsample_expand_ratios[stage_num - 1] if stride == 2
                else self.expand_ratio,
                fusedmbconv=self.fused_conv,
                expand_groups=self.expand_groups,
                norm_layer=nn.BatchNorm2D,
                act_layer=self.act_layer,
            )
            blocks.append(ResidualBlock(block, nn.Identity() if stride == 1 else None,
                                        dpr[i]))
            in_channels = width
        return blocks, in_channels

    def _build_attention_stage(self, in_channels, width, depth, stage_num, dpr):
        downsample = self.build_local_block(
            in_channels=in_channels,
            out_channels=width,
            stride=2,
            expand_ratio=self.downsample_expand_ratios[stage_num - 1],
            fusedmbconv=self.fused_downsample,
            expand_groups=self.expand_groups,
            norm_layer=nn.BatchNorm2D,
            act_layer=self.act_layer,
        )
        in_channels = width
        blocks = [ResidualBlock(downsample, None)]
        for i in range(depth):
            blocks.append(
                CPUBoneBlock(
                    in_channels=in_channels,
                    expand_ratio=self.expand_ratio,
                    act_layer=self.act_layer,
                    fused_conv=self.fused_conv,
                    expand_groups=self.expand_groups,
                    att_stride=2 if stage_num == 3 else 1,
                    mlp_ratio=self.attn_mlp_ratio,
                    small_kernels=self.small_kernels,
                    attn_upsample=self.attn_upsample,
                    proj_drop=self.proj_drop_rate,
                    drop_path=dpr[i],
                    local_mbconv_norm=self.local_mbconv_norm,
                )
            )
        return blocks, in_channels

    @staticmethod
    def build_local_block(in_channels, out_channels, stride, expand_ratio, norm_layer,
                          act_layer, fusedmbconv=False, expand_groups=1, kernel_size=3):
        if fusedmbconv:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(False, False),
                kernel_size=kernel_size,
                expand_groups=expand_groups,
                norm_layer=(norm_layer, norm_layer),
                act_layer=(act_layer, None),
            )
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                expand_groups=expand_groups,
                use_bias=(False, False, False),
                norm_layer=(None, None, norm_layer),
                act_layer=(act_layer, act_layer, None),
            )
        return block

    def reset_classifier(self, num_classes, global_pool=None):
        if global_pool is not None:
            _check_global_pool(global_pool)
            self.global_pool = global_pool
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x):
        return self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model):
    """Adapt legacy checkpoints; timm native checkpoints pass through."""
    if 'stem.0.conv.weight' in state_dict:
        return state_dict

    sd = remap_legacy_state_dict(state_dict)
    if getattr(model, 'local_mbconv_norm', None) == 'all':
        bias_keys = [k for k in sd if k.endswith((
            '.inverted_conv.conv.bias', '.depth_conv.conv.bias',
            '.inverted_conv.conv.1.bias', '.depth_conv.conv.1.bias',
        ))]
        for k in bias_keys:
            norm_key = k.replace('.conv.1.bias', '.norm.running_mean')
            norm_key = norm_key.replace('.conv.bias', '.norm.running_mean')
            if norm_key in sd:
                sd[norm_key] = sd[norm_key] - sd[k]
            del sd[k]
    return sd


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    from ppcls.utils import save_load

    if pretrained is False:
        pass
    elif pretrained is True:
        save_load.load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        save_load.load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def _create_cpubone(arch_args, variant, pretrained=False, **kwargs):
    """class_num is framework-reserved; explicit kwargs override variant defaults."""
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model = CPUBone(**dict(arch_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS[variant])
    return model


# All released checkpoints use the fastit/grouping=2/smallk_only_lasts/
# lose_transpose combination, i.e. the flags below.
_ARCH_ARGS = dict(
    fused_conv=True,
    fused_downsample=True,
    attn_mlp_ratio=4,
    expand_groups=2,
    small_kernels=True,
    attn_upsample="nearest",
)


def CPUBone_nano(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(_ARCH_ARGS, width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 1, 2])
    return _create_cpubone(model_args, "CPUBone_nano", pretrained=pretrained, **kwargs)


def CPUBone_t0(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(_ARCH_ARGS, width_list=[12, 24, 48, 96, 192], depth_list=[0, 1, 1, 2, 3])
    return _create_cpubone(model_args, "CPUBone_t0", pretrained=pretrained, **kwargs)


def CPUBone_s0(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(_ARCH_ARGS, width_list=[14, 28, 56, 112, 224], depth_list=[0, 1, 1, 2, 3])
    return _create_cpubone(model_args, "CPUBone_s0", pretrained=pretrained, **kwargs)


def CPUBone_b0_bfrobust(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        _ARCH_ARGS,
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 3, 4],
        local_mbconv_norm='all',
    )
    return _create_cpubone(model_args, "CPUBone_b0_bfrobust", pretrained=pretrained, **kwargs)


def _cpubone_b1_args(local_mbconv_norm='all'):
    return dict(
        _ARCH_ARGS,
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 5, 5],
        downsample_expand_ratios=(6, 6, 6, 6),
        local_mbconv_norm=local_mbconv_norm,
    )


def CPUBone_b1_bfrobust(pretrained=False, use_ssld=False, **kwargs):
    model_args = _cpubone_b1_args()
    return _create_cpubone(model_args, "CPUBone_b1_bfrobust", pretrained=pretrained, **kwargs)


def CPUBone_b1_dwnorm(pretrained=False, use_ssld=False, **kwargs):
    model_args = _cpubone_b1_args(local_mbconv_norm='depth_proj')
    return _create_cpubone(model_args, "CPUBone_b1_dwnorm", pretrained=pretrained, **kwargs)


def CPUBone_b2_bfrobust(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        _ARCH_ARGS,
        width_list=[20, 40, 80, 160, 320],
        depth_list=[0, 1, 1, 6, 6],
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
        drop_path_rate=0.1,
        local_mbconv_norm='all',
    )
    return _create_cpubone(model_args, "CPUBone_b2_bfrobust", pretrained=pretrained, **kwargs)


def CPUBone_b2pt5_dwnorm(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        _ARCH_ARGS,
        width_list=[24, 48, 96, 192, 384],
        depth_list=[0, 1, 1, 6, 6],
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
        local_mbconv_norm='depth_proj',
    )
    return _create_cpubone(model_args, "CPUBone_b2pt5_dwnorm", pretrained=pretrained, **kwargs)


def CPUBone_b3(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        _ARCH_ARGS,
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 3, 6, 6],
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_cpubone(model_args, "CPUBone_b3", pretrained=pretrained, **kwargs)
