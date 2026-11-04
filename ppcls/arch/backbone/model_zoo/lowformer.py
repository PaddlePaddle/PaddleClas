# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

# reference: https://github.com/altair199797/LowFormer

import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain
from ..base.theseus_layer import TheseusLayer
from .efficientvit import ConvNormAct, FusedMBConv, MBConv
from .vision_transformer import DropPath, trunc_normal_, zeros_

__all__ = [
    "LowFormer_b0",
    "LowFormer_b1",
    "LowFormer_b15",
    "LowFormer_b3",
    "LowFormer_e1",
    "LowFormer_e2",
    "LowFormer_e3",
]

MODEL_URLS = {
    "LowFormer_b0": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_b0.pdparams"
    ),
    "LowFormer_b1": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_b1.pdparams"
    ),
    "LowFormer_b15": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_b15.pdparams"
    ),
    "LowFormer_b3": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_b3.pdparams"
    ),
    "LowFormer_e1": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_e1.pdparams"
    ),
    "LowFormer_e2": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_e2.pdparams"
    ),
    "LowFormer_e3": (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "LowFormer_e3.pdparams"
    ),
}


def calculate_drop_path_rates(drop_path_rate, total):
    # per-block linear ramp, equals timm calculate_drop_path_rates for an int depth
    if total <= 1:
        return [0.0] * max(total, 0)
    return [drop_path_rate * i / (total - 1) for i in range(total)]


def _local_block(
    block_cls,
    in_channels,
    out_channels,
    kernel_size=3,
    stride=1,
    expand_ratio=6,
    use_bias=False,
    norm_layer=nn.BatchNorm2D,
    act_layer=nn.ReLU6,
):
    """Build an MBConv / FusedMBConv from LowFormer's scalar block config.

    EfficientViT's blocks take one norm / act / bias entry per inner conv, while LowFormer
    always uses the same pattern, so the tuples are filled in here: FusedMBConv normalises
    both convs, MBConv normalises only the projection and applies activation to the two
    expanded convs without normalising them.
    """
    args = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        expand_ratio=expand_ratio,
    )
    if block_cls is FusedMBConv:
        return block_cls(
            use_bias=(use_bias, False),
            norm_layer=(norm_layer, norm_layer),
            act_layer=(act_layer, None),
            **args,
        )
    return block_cls(
        use_bias=(use_bias, use_bias, False),
        norm_layer=(None, None, norm_layer),
        act_layer=(act_layer, act_layer, None),
        **args,
    )


class ResidualBlock(nn.Layer):
    def __init__(self, main, shortcut=None, drop_path=0.0):
        super().__init__()
        self.main = main
        self.shortcut = shortcut
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        if self.shortcut is None:
            return self.main(x)
        return self.drop_path(self.main(x)) + self.shortcut(x)


class ConvAttention(nn.Layer):
    """Low-frequency / conv-projected attention with strided downsample and learned upsample.

    The attention is written out by hand: upstream only dispatches to fused SDPA for fp16 /
    bf16, and in fp32 it runs exactly this sequence.
    """

    def __init__(
        self,
        input_dim,
        head_dim_mul=0.5,
        att_stride=4,
        att_kernel=7,
        fuse_out_proj=False,
    ):
        super().__init__()
        self.num_heads = int(max(1, (input_dim * head_dim_mul) // 30))
        self.head_dim = int((input_dim // self.num_heads) * head_dim_mul)
        self.num_keys = 3
        self.scale = self.head_dim**-0.5
        self.att_stride = att_stride

        total_dim = int(self.head_dim * self.num_heads * self.num_keys)

        self.conv_proj = ConvNormAct(
            input_dim,
            input_dim,
            kernel_size=att_kernel,
            stride=att_stride,
            groups=input_dim,
            norm_layer=nn.BatchNorm2D,
            act_layer=None,
        )
        self.pwise = nn.Conv2D(
            input_dim, total_dim, kernel_size=1, stride=1, padding=0, bias_attr=False
        )

        self.o_proj_inpdim = self.head_dim * self.num_heads
        # With fuse_out_proj the output projection is folded into the upsampling module below, which
        # then maps o_proj_inpdim -> input_dim instead of being a depthwise / parameter-free upsample.
        if fuse_out_proj:
            self.o_proj = nn.Identity()
            if att_stride == 1:
                self.upsampling = nn.Conv2DTranspose(
                    self.o_proj_inpdim, input_dim, kernel_size=3, stride=1, padding=1
                )
            else:
                self.upsampling = nn.Conv2DTranspose(
                    self.o_proj_inpdim,
                    input_dim,
                    kernel_size=att_stride * 2,
                    stride=att_stride,
                    padding=att_stride // 2,
                )
        else:
            self.o_proj = nn.Conv2D(
                self.o_proj_inpdim, input_dim, kernel_size=1, stride=1, padding=0
            )
            if att_stride == 1:
                self.upsampling = nn.Conv2DTranspose(
                    input_dim,
                    input_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=input_dim,
                )
            else:
                self.upsampling = nn.Conv2DTranspose(
                    input_dim,
                    input_dim,
                    kernel_size=att_stride * 2,
                    stride=att_stride,
                    padding=att_stride // 2,
                    groups=input_dim,
                )

    def forward(self, x):
        H, W = x.shape[-2], x.shape[-1]

        xout = self.conv_proj(x)
        xout = self.pwise(xout)

        N, _, h, w = xout.shape
        qkv = xout.reshape([N, self.num_heads, self.num_keys * self.head_dim, h * w])
        qkv = qkv.transpose([0, 1, 3, 2])
        q, k, v = qkv.chunk(3, axis=3)

        q = q * self.scale
        attn = q @ k.transpose([0, 1, 3, 2])
        attn = F.softmax(attn, axis=-1)
        values = attn @ v
        o = self.o_proj(
            values.transpose([0, 1, 3, 2]).reshape([N, self.o_proj_inpdim, h, w])
        )

        o = self.upsampling(o)
        # Upsampling can overshoot after same-padding, crop to the input spatial size
        return o[:, :, :H, :W]


class LowFormerBlock(nn.Layer):
    """Attention (context) and MLP branches followed by a local conv branch, all with identity residuals."""

    def __init__(
        self,
        in_channels,
        expand_ratio=4,
        norm_layer=nn.BatchNorm2D,
        act_layer=nn.Hardswish,
        fused_conv=False,
        attn=True,
        attn_mlp=True,
        attn_mlp_ratio=4,
        att_stride=1,
        proj_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        att_kernel = 5 if att_stride > 1 else 3

        if attn:
            attn_module = ConvAttention(
                input_dim=in_channels,
                att_stride=att_stride,
                att_kernel=att_kernel,
                head_dim_mul=0.5,
                fuse_out_proj=fused_conv,
            )
            if attn_mlp:
                attn_module = nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=in_channels), attn_module
                )
            self.attn = ResidualBlock(attn_module, nn.Identity(), drop_path)
        else:
            self.attn = nn.Identity()

        if attn_mlp:
            self.mlp = ResidualBlock(
                nn.Sequential(
                    nn.GroupNorm(num_groups=1, num_channels=in_channels),
                    nn.Conv2D(in_channels, in_channels * attn_mlp_ratio, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2D(in_channels * attn_mlp_ratio, in_channels, kernel_size=1),
                    nn.Dropout(proj_drop),
                ),
                nn.Identity(),
                drop_path,
            )
        else:
            self.mlp = nn.Identity()

        block_cls = FusedMBConv if fused_conv and in_channels < 256 else MBConv
        local_module = _local_block(
            block_cls,
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=True,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.local = ResidualBlock(local_module, nn.Identity(), drop_path)

    def forward(self, x):
        x = self.attn(x)
        x = self.mlp(x)
        x = self.local(x)
        return x


class LowFormer(TheseusLayer):
    def __init__(
        self,
        width_list,
        depth_list,
        in_chans=3,
        num_classes=1000,
        global_pool="avg",
        head_widths=(1536, 1600),
        drop_rate=0.0,
        proj_drop_rate=0.0,
        drop_path_rate=0.0,
        expand_ratio=4,
        norm_layer=nn.BatchNorm2D,
        act_layer=nn.Hardswish,
        fused_conv=True,
        attn=True,
        attn_mlp=True,
        attn_mlp_ratio=4,
        stem_expand_ratio=2,
        downsample_expand_ratios=None,
    ):
        super().__init__()
        num_stages = len(width_list) - 1
        if downsample_expand_ratios is None:
            downsample_expand_ratios = (expand_ratio,) * num_stages
        assert len(downsample_expand_ratios) == num_stages
        downsample_expand_ratios = tuple(downsample_expand_ratios)

        self.num_classes = num_classes
        self.in_chans = in_chans
        self.drop_rate = drop_rate

        # stochastic depth: linear ramp of drop rates across all blocks (downsample blocks have no
        # shortcut and ignore theirs)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depth_list))
        block_cls = FusedMBConv if fused_conv else MBConv

        # stem: stride-2 conv, then `depth_list[0]` local blocks at the stem width
        stem_blocks = [
            ConvNormAct(
                in_channels=in_chans,
                out_channels=width_list[0],
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
        ]
        in_channels = width_list[0]
        for i in range(depth_list[0]):
            block = _local_block(
                block_cls,
                in_channels=in_channels,
                out_channels=in_channels,
                stride=1,
                expand_ratio=stem_expand_ratio,
                use_bias=False,
                norm_layer=norm_layer,
                act_layer=act_layer,
            )
            stem_blocks.append(ResidualBlock(block, nn.Identity(), dpr[i]))
        self.stem = nn.Sequential(*stem_blocks)

        # stages 1-4: early stages use plain conv blocks, later stages add attention
        stages = []
        block_idx = depth_list[0]
        for stage_num, (width, depth) in enumerate(
            zip(width_list[1:], depth_list[1:]), start=1
        ):
            stage_dpr = dpr[block_idx : block_idx + depth]
            block_idx += depth
            blocks = []
            if stage_num >= 3:
                downsample = _local_block(
                    block_cls,
                    in_channels=in_channels,
                    out_channels=width,
                    stride=2,
                    expand_ratio=downsample_expand_ratios[stage_num - 1],
                    use_bias=False,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                blocks.append(ResidualBlock(downsample, None))
                in_channels = width
                for i in range(depth):
                    blocks.append(
                        LowFormerBlock(
                            in_channels=in_channels,
                            expand_ratio=expand_ratio,
                            norm_layer=norm_layer,
                            act_layer=act_layer,
                            fused_conv=fused_conv,
                            attn=attn,
                            attn_mlp=attn_mlp,
                            attn_mlp_ratio=attn_mlp_ratio,
                            att_stride=2 if stage_num == 3 else 1,
                            proj_drop=proj_drop_rate,
                            drop_path=stage_dpr[i],
                        )
                    )
            elif depth > 0:
                for i in range(depth):
                    stride = 2 if i == 0 else 1
                    block = _local_block(
                        block_cls,
                        in_channels=in_channels,
                        out_channels=width,
                        stride=stride,
                        expand_ratio=(
                            downsample_expand_ratios[stage_num - 1]
                            if stride == 2
                            else expand_ratio
                        ),
                        use_bias=False,
                        norm_layer=norm_layer,
                        act_layer=act_layer,
                    )
                    blocks.append(
                        ResidualBlock(
                            block, nn.Identity() if stride == 1 else None, stage_dpr[i]
                        )
                    )
                    in_channels = width
            # a zero-depth stage contributes no blocks
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # head
        self.num_features = width_list[-1]
        self.head_hidden_size = head_widths[-1]
        self._set_global_pool(global_pool)
        self.in_conv = ConvNormAct(
            in_channels=width_list[-1],
            out_channels=head_widths[0],
            kernel_size=1,
            norm_layer=norm_layer,
            act_layer=act_layer,
        )
        self.pre_classifier = nn.Sequential(
            nn.Linear(head_widths[0], head_widths[1], bias_attr=False),
            nn.LayerNorm(head_widths[1]),
            act_layer(),
        )
        self.classifier = (
            nn.Linear(head_widths[1], num_classes) if num_classes > 0 else nn.Identity()
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight)
            if module.bias is not None:
                zeros_(module.bias)

    def _set_global_pool(self, global_pool):
        assert global_pool in (
            "",
            "avg",
        ), "LowFormer only supports average or disabled pooling"
        self.global_pool_type = global_pool
        self.global_pool = (
            nn.AdaptiveAvgPool2D(output_size=1) if global_pool else nn.Identity()
        )
        self.flatten = nn.Flatten(start_axis=1) if global_pool else nn.Identity()

    def reset_classifier(self, num_classes, global_pool=None):
        was_training = self.training
        self.num_classes = num_classes
        if global_pool is not None:
            self._set_global_pool(global_pool)
            self.global_pool.train(was_training)
        self.classifier = (
            nn.Linear(self.head_hidden_size, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.classifier.train(was_training)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        return x

    def forward_head(self, x, pre_logits=False):
        x = self.in_conv(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        if not self.global_pool_type:
            # Keep the pretrained Linear/LayerNorm parameter shapes while applying them channel-wise.
            x = x.transpose([0, 2, 3, 1])
        x = self.pre_classifier(x)
        if self.drop_rate:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if not pre_logits:
            x = self.classifier(x)
        if not self.global_pool_type:
            x = x.transpose([0, 3, 1, 2])
        return x

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


def _create_lowformer(arch_args, variant, pretrained=False, use_ssld=False, **kwargs):
    """class_num is framework-reserved; explicit kwargs override variant defaults."""
    if "class_num" in kwargs:
        kwargs["num_classes"] = kwargs.pop("class_num")
    model = LowFormer(**dict(arch_args, **kwargs))
    _load_pretrained(pretrained, model, MODEL_URLS[variant], use_ssld=use_ssld)
    return model


def LowFormer_b0(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 3, 4],
    )
    return _create_lowformer(
        model_args, "LowFormer_b0", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_b1(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[16, 32, 64, 128, 256],
        depth_list=[0, 1, 1, 5, 5],
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_b1", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_b15(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[20, 40, 80, 160, 320],
        depth_list=[0, 1, 1, 6, 6],
        head_widths=(2304, 2560),
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_b15", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_b3(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 3, 6, 6],
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_b3", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_e1(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[20, 40, 80, 160, 320],
        depth_list=[0, 1, 1, 4, 4],
        head_widths=(2304, 2560),
        attn=False,
        attn_mlp=False,
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_e1", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_e2(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 3, 4, 4],
        attn=False,
        attn_mlp=False,
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_e2", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )


def LowFormer_e3(pretrained=False, use_ssld=False, **kwargs):
    model_args = dict(
        width_list=[32, 64, 128, 256, 512],
        depth_list=[1, 2, 3, 6, 6],
        attn_mlp=False,
        stem_expand_ratio=4,
        downsample_expand_ratios=(6, 6, 6, 6),
    )
    return _create_lowformer(
        model_args, "LowFormer_e3", pretrained=pretrained, use_ssld=use_ssld, **kwargs
    )
