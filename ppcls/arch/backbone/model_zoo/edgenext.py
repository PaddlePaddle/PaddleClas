# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import math
from functools import partial

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Constant, TruncatedNormal

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "EdgeNeXt_XX_Small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/edgenext_xx_small.pdparams",
    "EdgeNeXt_X_Small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/edgenext_x_small.pdparams",
    "EdgeNeXt_Small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/edgenext_small.pdparams",
    "EdgeNeXt_Base": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/edgenext_base.pdparams",
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=0.02)
zeros_ = Constant(value=0.0)
ones_ = Constant(value=1.0)


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels, epsilon=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[num_channels], default_initializer=ones_
        )
        self.bias = self.create_parameter(
            shape=[num_channels], default_initializer=zeros_
        )
        self.epsilon = epsilon

    def forward(self, x):
        u = x.mean(axis=1, keepdim=True)
        s = (x - u).pow(2).mean(axis=1, keepdim=True)
        inv_s = paddle.pow(s + self.epsilon, -0.5)
        x = (x - u) * inv_s
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Mlp(nn.Layer):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PositionalEncodingFourier(nn.Layer):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000.0):
        super().__init__()
        self.token_projection = nn.Conv2D(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, shape):
        dtype = self.token_projection.weight.dtype
        inv_mask = paddle.ones(shape, dtype="bool")
        y_embed = inv_mask.cumsum(axis=1, dtype="float32")
        x_embed = inv_mask.cumsum(axis=2, dtype="float32")

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = paddle.arange(self.hidden_dim, dtype="int64").cast("float32")
        dim_t = self.temperature ** (2 * paddle.floor(dim_t / 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = paddle.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), axis=4
        ).flatten(3)
        pos_y = paddle.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), axis=4
        ).flatten(3)

        pos = paddle.concat((pos_y, pos_x), axis=3).transpose([0, 3, 1, 2])
        pos = self.token_projection(pos.cast(dtype))

        return pos


class ConvBlock(nn.Layer):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=7,
        stride=1,
        conv_bias=True,
        expand_ratio=4,
        ls_init_value=1e-6,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        act_layer=nn.GELU,
        drop_path=0.0,
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.shortcut_after_dw = stride > 1 or dim != dim_out
        pad = (kernel_size - 1) // 2
        self.conv_dw = nn.Conv2D(
            dim,
            dim_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=dim,
            bias_attr=conv_bias,
        )

        self.norm = norm_layer(dim_out)
        self.mlp = Mlp(dim_out, int(expand_ratio * dim_out), act_layer=act_layer)
        if ls_init_value > 0:
            self.gamma = self.create_parameter(
                shape=[dim_out], default_initializer=Constant(value=ls_init_value)
            )
        else:
            self.gamma = None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.shortcut_after_dw:
            shortcut = x
        x = x.transpose([0, 2, 3, 1])
        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])

        x = shortcut + self.drop_path(x)
        return x


class CrossCovarianceAttn(nn.Layer):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = self.create_parameter(
            shape=[num_heads, 1, 1], default_initializer=ones_
        )

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape([B, N, 3, self.num_heads, -1])
            .transpose([2, 0, 3, 4, 1])
        )
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        q_norm = paddle.nn.functional.normalize(q, axis=-1)
        k_norm = paddle.nn.functional.normalize(k, axis=-1)
        attn = paddle.bmm(
            q_norm.reshape([-1, q_norm.shape[-2], q_norm.shape[-1]]),
            k_norm.reshape([-1, k_norm.shape[-2], k_norm.shape[-1]]).transpose(
                [0, 2, 1]
            ),
        )
        attn = attn.reshape([B, self.num_heads, q.shape[-2], k.shape[-2]])
        attn = attn * self.temperature
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = paddle.bmm(
            attn.reshape([-1, attn.shape[-2], attn.shape[-1]]),
            v.reshape([-1, v.shape[-2], v.shape[-1]]),
        )
        x = x.reshape([B, self.num_heads, x.shape[-2], x.shape[-1]])

        x = x.transpose([0, 3, 1, 2]).reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def no_weight_decay(self):
        return {"temperature"}


class SplitTransposeBlock(nn.Layer):
    def __init__(
        self,
        dim,
        num_scales=1,
        num_heads=8,
        expand_ratio=4,
        use_pos_emb=True,
        conv_bias=True,
        qkv_bias=True,
        ls_init_value=1e-6,
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
        act_layer=nn.GELU,
        drop_path=0.0,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        width = max(
            int(math.ceil(dim / num_scales)), int(math.floor(dim // num_scales))
        )
        self.width = width
        self.num_scales = max(1, num_scales - 1)

        convs = []
        for i in range(self.num_scales):
            convs.append(
                nn.Conv2D(
                    width,
                    width,
                    kernel_size=3,
                    padding=1,
                    groups=width,
                    bias_attr=conv_bias,
                )
            )
        self.convs = nn.LayerList(convs)

        self.pos_embd = None
        if use_pos_emb:
            self.pos_embd = PositionalEncodingFourier(dim=dim)

        self.norm_xca = norm_layer(dim)
        if ls_init_value > 0:
            self.gamma_xca = self.create_parameter(
                shape=[dim], default_initializer=Constant(value=ls_init_value)
            )
        else:
            self.gamma_xca = None

        self.xca = CrossCovarianceAttn(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm = norm_layer(dim)
        self.mlp = Mlp(dim, int(expand_ratio * dim), act_layer=act_layer)
        if ls_init_value > 0:
            self.gamma = self.create_parameter(
                shape=[dim], default_initializer=Constant(value=ls_init_value)
            )
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        shortcut = x

        num_chunks = len(self.convs) + 1
        total = x.shape[1]
        split_size = math.ceil(total / num_chunks)
        split_sizes = [split_size] * (total // split_size)
        if total % split_size != 0:
            split_sizes.append(total % split_size)
        spx = paddle.split(x, split_sizes, axis=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = paddle.concat(spo, axis=1)

        B, C, H, W = x.shape
        x = x.reshape([B, C, H * W]).transpose([0, 2, 1])
        if self.pos_embd is not None:
            pos_encoding = (
                self.pos_embd((B, H, W))
                .reshape([B, -1, x.shape[1]])
                .transpose([0, 2, 1])
            )
            x = x + pos_encoding

        xca_input = self.norm_xca(x)
        xca_out = self.xca(xca_input)
        if self.gamma_xca is not None:
            xca_out = self.gamma_xca * xca_out
        x = x + self.drop_path(xca_out)

        x = x.reshape([B, H, W, C])

        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])

        x = shortcut + self.drop_path(x)
        return x


class EdgeNeXtStage(nn.Layer):
    def __init__(
        self,
        in_chs,
        out_chs,
        stride=2,
        depth=2,
        num_global_blocks=1,
        num_heads=4,
        scales=2,
        kernel_size=7,
        expand_ratio=4,
        use_pos_emb=False,
        downsample_block=False,
        conv_bias=True,
        ls_init_value=1.0,
        drop_path_rates=None,
        norm_layer=LayerNorm2d,
        norm_layer_cl=partial(nn.LayerNorm, epsilon=1e-6),
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.grad_checkpointing = False

        if downsample_block or stride == 1:
            self.downsample = nn.Identity()
        else:
            self.downsample = nn.Sequential(
                norm_layer(in_chs),
                nn.Conv2D(
                    in_chs, out_chs, kernel_size=2, stride=2, bias_attr=conv_bias
                ),
            )
            in_chs = out_chs

        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        stage_blocks = []
        for i in range(depth):
            if i < depth - num_global_blocks:
                stage_blocks.append(
                    ConvBlock(
                        dim=in_chs,
                        dim_out=out_chs,
                        stride=stride if downsample_block and i == 0 else 1,
                        conv_bias=conv_bias,
                        kernel_size=kernel_size,
                        expand_ratio=expand_ratio,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            else:
                stage_blocks.append(
                    SplitTransposeBlock(
                        dim=in_chs,
                        num_scales=scales,
                        num_heads=num_heads,
                        expand_ratio=expand_ratio,
                        use_pos_emb=use_pos_emb,
                        conv_bias=conv_bias,
                        ls_init_value=ls_init_value,
                        drop_path=drop_path_rates[i],
                        norm_layer=norm_layer_cl,
                        act_layer=act_layer,
                    )
                )
            in_chs = out_chs
        self.blocks = nn.Sequential(*stage_blocks)

    def forward(self, x):
        x = self.downsample(x)
        x = self.blocks(x)
        return x


def _calc_drop_path_rates(drop_path_rate, depths, stagewise=True):
    if stagewise:
        total_depth = sum(depths)
        dp_rates = []
        rates = list(paddle.linspace(0, drop_path_rate, total_depth + 1).numpy())
        cur = 0
        for d in depths:
            stage_rates = rates[cur : cur + d]
            dp_rates.append(stage_rates)
            cur += d
        return dp_rates
    else:
        return [drop_path_rate] * len(depths)


class EdgeNeXt(nn.Layer):
    """EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture.

    Original: EdgeNeXt in timm edgenext.py

    Args:
        in_chans (int): Number of input image channels. Default: 3
        class_num (int): Number of classes for classification head. Default: 1000
        dims (tuple): Feature dimension at each stage.
        depths (tuple): Number of blocks at each stage.
        global_block_counts (tuple): Number of global (transformer) blocks per stage.
        kernel_sizes (tuple): Kernel sizes for depthwise convolutions per stage.
        heads (tuple): Number of attention heads per stage.
        d2_scales (tuple): Number of scales for split transpose blocks per stage.
        use_pos_emb (tuple): Whether to use positional encoding per stage.
        ls_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        expand_ratio (float): Expand ratio for MLP. Default: 4.
        downsample_block (bool): Whether to use block-based downsampling. Default: False.
        conv_bias (bool): Whether to use bias in convolutions. Default: True.
        stem_type (str): Type of stem ('patch' or 'overlap'). Default: 'patch'.
        head_norm_first (bool): Whether to apply norm before head. Default: False.
        act_layer: Activation layer. Default: nn.GELU.
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        drop_rate (float): Dropout rate. Default: 0.
    """

    def __init__(
        self,
        in_chans=3,
        class_num=1000,
        dims=(24, 48, 88, 168),
        depths=(3, 3, 9, 3),
        global_block_counts=(0, 1, 1, 1),
        kernel_sizes=(3, 5, 7, 9),
        heads=(8, 8, 8, 8),
        d2_scales=(2, 2, 3, 4),
        use_pos_emb=(False, True, False, False),
        ls_init_value=1e-6,
        head_init_scale=1.0,
        expand_ratio=4,
        downsample_block=False,
        conv_bias=True,
        stem_type="patch",
        head_norm_first=False,
        act_layer=nn.GELU,
        drop_path_rate=0.0,
        drop_rate=0.0,
    ):
        super().__init__()
        self.num_classes = class_num
        self.drop_rate = drop_rate

        norm_layer = partial(LayerNorm2d, epsilon=1e-6)
        norm_layer_cl = partial(nn.LayerNorm, epsilon=1e-6)
        self.feature_info = []

        assert stem_type in ("patch", "overlap")
        if stem_type == "patch":
            self.stem = nn.Sequential(
                nn.Conv2D(
                    in_chans, dims[0], kernel_size=4, stride=4, bias_attr=conv_bias
                ),
                norm_layer(dims[0]),
            )
        else:
            self.stem = nn.Sequential(
                nn.Conv2D(
                    in_chans,
                    dims[0],
                    kernel_size=9,
                    stride=4,
                    padding=9 // 2,
                    bias_attr=conv_bias,
                ),
                norm_layer(dims[0]),
            )

        curr_stride = 4
        stages = []
        dp_rates = _calc_drop_path_rates(drop_path_rate, depths, stagewise=True)
        in_chs = dims[0]
        for i in range(4):
            stride = 2 if curr_stride == 2 or i > 0 else 1
            curr_stride *= stride
            stages.append(
                EdgeNeXtStage(
                    in_chs=in_chs,
                    out_chs=dims[i],
                    stride=stride,
                    depth=depths[i],
                    num_global_blocks=global_block_counts[i],
                    num_heads=heads[i],
                    drop_path_rates=dp_rates[i],
                    scales=d2_scales[i],
                    expand_ratio=expand_ratio,
                    kernel_size=kernel_sizes[i],
                    use_pos_emb=use_pos_emb[i],
                    ls_init_value=ls_init_value,
                    downsample_block=downsample_block,
                    conv_bias=conv_bias,
                    norm_layer=norm_layer,
                    norm_layer_cl=norm_layer_cl,
                    act_layer=act_layer,
                )
            )
            in_chs = dims[i]
            self.feature_info += [
                dict(num_chs=in_chs, reduction=curr_stride, module=f"stages.{i}")
            ]

        self.stages = nn.Sequential(*stages)

        self.num_features = dims[-1]
        if head_norm_first:
            self.norm_pre = norm_layer(self.num_features)

            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Flatten(1),
                nn.Dropout(drop_rate),
                nn.Linear(self.num_features, class_num),
            )
        else:
            self.norm_pre = nn.Identity()
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2D(1),
                nn.Flatten(1),
                LayerNorm2d(self.num_features),
                nn.Dropout(drop_rate),
                nn.Linear(self.num_features, class_num),
            )

        self.apply(partial(_init_weights, head_init_scale=head_init_scale))

    def forward_features(self, x):
        x = self.stem(x)
        x = self.stages(x)
        x = self.norm_pre(x)
        return x

    def forward_head(self, x):
        if isinstance(self.head, nn.Sequential):
            x = self.head[0](x)  # AdaptiveAvgPool2D
            x = self.head[1](x)  # Flatten
            if len(self.head) > 3 and isinstance(self.head[2], LayerNorm2d):
                x = x.reshape([x.shape[0], self.num_features, 1, 1])
                x = self.head[2](x)
                x = x.reshape([x.shape[0], self.num_features])
                x = self.head[3](x)  # Dropout
                x = self.head[4](x)  # Linear
            else:
                x = self.head[2](x)  # Dropout
                x = self.head[3](x)  # Linear
        else:
            x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _init_weights(module, name=None, head_init_scale=1.0):
    if isinstance(module, nn.Conv2D):
        trunc_normal_(module.weight)
        if module.bias is not None:
            zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        trunc_normal_(module.weight)
        if module.bias is not None:
            zeros_(module.bias)
            if name and "head." in name:
                module.weight.set_value(module.weight * head_init_scale)
                module.bias.set_value(module.bias * head_init_scale)


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


def EdgeNeXt_XX_Small(pretrained=False, use_ssld=False, **kwargs):
    model = EdgeNeXt(
        depths=(2, 2, 6, 2), dims=(24, 48, 88, 168), heads=(4, 4, 4, 4), **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["EdgeNeXt_XX_Small"], use_ssld=use_ssld
    )
    return model


def EdgeNeXt_X_Small(pretrained=False, use_ssld=False, **kwargs):
    model = EdgeNeXt(
        depths=(3, 3, 9, 3), dims=(32, 64, 100, 192), heads=(4, 4, 4, 4), **kwargs
    )
    _load_pretrained(
        pretrained, model, MODEL_URLS["EdgeNeXt_X_Small"], use_ssld=use_ssld
    )
    return model


def EdgeNeXt_Small(pretrained=False, use_ssld=False, **kwargs):
    model = EdgeNeXt(depths=(3, 3, 9, 3), dims=(48, 96, 160, 304), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EdgeNeXt_Small"], use_ssld=use_ssld)
    return model


def EdgeNeXt_Base(pretrained=False, use_ssld=False, **kwargs):
    model = EdgeNeXt(depths=[3, 3, 9, 3], dims=[80, 160, 288, 584], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["EdgeNeXt_Base"], use_ssld=use_ssld)
    return model
