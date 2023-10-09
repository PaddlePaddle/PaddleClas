# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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

# Code was based on https://github.com/BR-IDL/PaddleViT/blob/develop/image_classification/CSwin/cswin.py
# reference: https://arxiv.org/abs/2107.00652

import copy
import numpy as np
import paddle
import paddle.nn as nn
from .vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "CSWinTransformer_tiny_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_tiny_224_pretrained.pdparams",
    "CSWinTransformer_small_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_small_224_pretrained.pdparams",
    "CSWinTransformer_base_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_base_224_pretrained.pdparams",
    "CSWinTransformer_large_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_large_224_pretrained.pdparams",
    "CSWinTransformer_base_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_base_384_pretrained.pdparams",
    "CSWinTransformer_large_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CSWinTransformer_large_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


class PatchEmbedding(nn.Layer):
    """CSwin Patch Embedding
    This patch embedding has a 7x7 conv + layernorm, the output tensor
    is reshaped to [Batch, H*W, embed_dim]. Note that the patch is applied
    by a conv with overlap (using patch_stride).
    Args:
        patch_stride: int, patch stride size, default: 4
        in_channels: int, number of channels of input image, default: 3
        embed_dim: int, output feature dimension, default: 96
    """

    def __init__(self, patch_stride=4, in_channels=3, embed_dim=96):
        super().__init__()
        self.patch_embed = nn.Conv2D(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=7,
            stride=patch_stride,
            padding=2)

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(
            x)  # [batch, embed_dim, h, w], h = w = image_size / 4
        x = x.flatten(start_axis=2, stop_axis=-1)  # [batch, embed_dim, h*w]
        x = x.transpose([0, 2, 1])  # [batch, h*w, embed_dim]
        x = self.norm(x)
        return x


class Mlp(nn.Layer):
    """ MLP module
    Impl using nn.Linear and activation is GELU, dropout is applied.
    Ops: fc -> act -> dropout -> fc -> dropout
    Attributes:
        fc1: nn.Linear
        fc2: nn.Linear
        act: GELU
        dropout1: dropout after fc1
        dropout2: dropout after fc2
    """

    def __init__(self, in_features, hidden_features, dropout):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


def img2windows(img, h_split, w_split):
    """Convert input tensor into split stripes
    Args:
        img: tensor, image tensor with shape [B, C, H, W]
        h_split: int, splits width in height direction
        w_split: int, splits width in width direction
    Returns:
        out: tensor, splitted image
    """
    B, C, H, W = img.shape
    out = img.reshape([B, C, H // h_split, h_split, W // w_split, w_split])
    out = out.transpose(
        [0, 2, 4, 3, 5, 1])  # [B, H//h_split, W//w_split, h_split, w_split, C]
    out = out.reshape([-1, h_split * w_split,
                       C])  # [B, H//h_split, W//w_split, h_split*w_split, C]
    return out


def windows2img(img_splits, h_split, w_split, img_h, img_w):
    """Convert splitted stripes back
    Args:
        img_splits: tensor, image tensor with shape [B, C, H, W]
        h_split: int, splits width in height direction
        w_split: int, splits width in width direction
        img_h: int, original tensor height
        img_w: int, original tensor width
    Returns:
        img: tensor, original tensor
    """
    B = paddle.to_tensor(img_splits.shape[0] //
                         (img_h // h_split * img_w // w_split), "int32")
    img = img_splits.reshape([
        B, img_h // h_split, img_w // w_split, h_split, w_split,
        img_splits.shape[-1]
    ])
    img = img.transpose(
        [0, 1, 3, 2, 4,
         5])  #[B,img_h//h_split, h_split, img_w//w_split, w_split,C]
    img = img.reshape(
        [B, img_h, img_w, img_splits.shape[-1]])  # [B, img_h, img_w, C]
    return img


class LePEAttention(nn.Layer):
    """Cross Shaped Window self-attention with Locally enhanced positional encoding"""

    def __init__(self,
                 dim,
                 resolution,
                 h_split=7,
                 w_split=7,
                 num_heads=8,
                 attention_dropout=0.,
                 dropout=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head**-0.5
        self.h_split = h_split
        self.w_split = w_split

        self.get_v = nn.Conv2D(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=dim)

        self.softmax = nn.Softmax(axis=-1)
        self.attn_dropout = nn.Dropout(attention_dropout)

    def im2cswin(self, x):
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        x = x.transpose([0, 2, 1])  # [B, C, H*W]
        x = x.reshape([B, C, H, W])  # [B, C, H, W]
        x = img2windows(x, self.h_split, self.w_split)
        x = x.reshape(
            [-1, self.h_split * self.w_split, self.num_heads, self.dim_head])
        x = x.transpose([0, 2, 1, 3])
        return x

    def get_lepe(self, x, func):
        """Locally Enhanced Positional Encoding (LePE)
        This module applies a depthwise conv on V and returns the lepe
        Args:
            x: tensor, the input tensor V
            func: nn.Layer, a depth wise conv of kernel 3 stride 1 and padding 1
        """
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        h_split = self.h_split
        w_split = self.w_split

        x = x.transpose([0, 2, 1])  # [B, C, H*W]
        x = x.reshape([B, C, H, W])  # [B, C, H, W]
        x = x.reshape([B, C, H // h_split, h_split, W // w_split, w_split])
        x = x.transpose(
            [0, 2, 4, 1, 3,
             5])  # [B, H//h_split, W//w_split, C, h_split, w_split]
        x = x.reshape(
            [-1, C, h_split,
             w_split])  # [B*(H//h_split)*(W//w_split), h_split, w_split]

        lepe = func(x)  # depth wise conv does not change shape
        #lepe = lepe.reshape([-1, self.num_heads, C // self.num_heads, h_split * w_split])
        lepe = lepe.reshape(
            [-1, self.num_heads, self.dim_head, h_split * w_split])
        lepe = lepe.transpose(
            [0, 1, 3, 2])  # [B, num_heads, h_spllit*w_split, dim_head]

        x = x.reshape([-1, self.num_heads, self.dim_head, h_split * w_split])
        x = x.transpose(
            [0, 1, 3, 2])  # [B, num_heads, h_split*wsplit, dim_head]
        return x, lepe

    def forward(self, q, k, v):
        B, HW, C = q.shape
        H = W = self.resolution
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)
        z = z + lepe
        z = z.transpose([0, 2, 1, 3])
        z = z.reshape([-1, self.h_split * self.w_split, C])

        z = windows2img(z, self.h_split, self.w_split, H, W)
        z = z.reshape([B, z.shape[1] * z.shape[2], C])
        return z


class CSwinBlock(nn.Layer):
    """CSwin Block
    CSwin block contains a LePE attention modual, a linear projection,
    a mlp layer, and related norms layers. In the first 3 stages, the
    LePE attention moduals used 2 branches, where horizontal and
    vertical split stripes are used for self attention and a concat
    op is applied to combine the outputs. The last stage does not
    have branche in LePE attention.
    Args:
        dim: int, input feature dimension
        input_resolution: int, input feature spatial size.
        num_heads: int, num of attention heads in current stage
        split_size: int, the split size in current stage
        mlp_ratio: float, mlp ratio, mlp_hidden_dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if set True, qkv projection will have bias, default: True
        qk_scale: float, if set, replace the orig qk_scale (dim_head ** -0.5), default: None
        dropout: float, dropout rate for linear projection, default: 0
        attention_dropout: float, dropout rate for attention, default: 0
        droppath: float, drop path rate, default: 0
        split_heads: bool, if True, split heads is applied (True for 1,2,3 stages), default: True
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 split_size=7,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.,
                 droppath=0.,
                 split_heads=True):
        super().__init__()
        self.dim = dim
        # NOTE: here assume image_h == imgae_w
        self.input_resolution = (input_resolution, input_resolution)
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.mlp_ratio = mlp_ratio
        self.split_size = split_size
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(
            dim, dim * 3, bias_attr=None if qkv_bias else False)
        self.attns = nn.LayerList()
        self.split_heads = split_heads

        num_branches = 2 if split_heads else 1
        if split_heads:  # first 3 stages
            splits = [self.input_resolution[0],
                      self.split_size]  # horizantal splits
        else:  # last stage
            splits = [self.input_resolution[0], self.input_resolution[0]]
        for _ in range(num_branches):
            attn = LePEAttention(
                dim=dim // num_branches,
                resolution=input_resolution,
                h_split=splits[0],
                w_split=splits[1],
                num_heads=num_heads // num_branches,
                qk_scale=qk_scale,
                attention_dropout=attention_dropout,
                dropout=dropout)
            self.attns.append(copy.deepcopy(attn))
            # switch splits from horizantal to vertical
            # NOTE: may need to change for different H and W
            splits[0], splits[1] = splits[1], splits[0]

        self.proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

    def chunk_qkv(self, x, chunks=1, axis=-1):
        x = x.chunk(chunks, axis=axis)
        return x

    def forward(self, x):
        H, W = self.input_resolution
        B, HW, C = x.shape
        # cswin attention
        h = x
        x = self.norm1(x)
        qkv = self.qkv(x).chunk(3, axis=-1)  # qkv is a tuple of [q, k, v]

        if self.split_heads:
            q, k, v = map(self.chunk_qkv, qkv,
                          (2, 2, 2))  # map requries list/tuple inputs
        else:
            q, k, v = map(lambda x: [x], qkv)

        if self.split_heads:  # first 3 stages
            h_attn = self.attns[0](q[0], k[0], v[0])
            w_attn = self.attns[1](q[1], k[1], v[1])
            attn = paddle.concat([h_attn, w_attn], axis=2)
        else:  # last stage
            attn = self.attns[0](q[0], k[0], v[0])
        attn = self.proj(attn)
        attn = self.drop_path(attn)
        x = h + attn
        # mlp + residual
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = h + x
        return x


class MergeBlock(nn.Layer):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=dim_in,
            out_channels=dim_out,
            kernel_size=3,
            stride=2,
            padding=1)
        self.norm = nn.LayerNorm(dim_out)

    def forward(self, x):
        B, HW, C = x.shape
        H = W = int(np.sqrt(HW))
        x = x.transpose([0, 2, 1])  # [B, C, HW]
        x = x.reshape([B, C, H, W])  # [B, C, H, W]
        x = self.conv(x)
        new_shape = [x.shape[0], x.shape[1],
                     x.shape[2] * x.shape[3]]  # [B, C', H*W]
        x = x.reshape(new_shape)  # [B, C', H*W]
        x = x.transpose([0, 2, 1])  # [B, H*W, C']
        x = self.norm(x)
        return x


class CSwinStage(nn.Layer):
    """ CSwin Stage, each stage contains multi blocks
    CSwin has 4 stages, the first 3 stages are using head split. The last
    stage does not have head split. There is a merge block between each
    2 stages.
    Args:
        dim: int, input feature dimension
        depth: int, number of blocks in current stage
        num_heads: int, num of attention heads in current stage
        split_size: int, the split size in current stage
        mlp_ratio: float, mlp ratio, mlp_hidden_dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if set True, qkv projection will have bias, default: True
        qk_scale: float, if set, replace the orig qk_scale (dim_head ** -0.5), default: None
        dropout: float, dropout rate for linear projection, default: 0
        attention_dropout: float, dropout rate for attention, default: 0
        droppath: float, drop path rate, default: 0
        last_stage: bool, if current stage is the last stage, default: False
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 split_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 last_stage=False):
        super().__init__()
        self.blocks = nn.LayerList()
        for i in range(depth):
            block = CSwinBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                split_size=split_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attention_dropout=attention_dropout,
                dropout=dropout,
                droppath=droppath[i]
                if isinstance(droppath, list) else droppath,
                split_heads=not last_stage)
            self.blocks.append(copy.deepcopy(block))
        # last stage does not need merge layer
        self.merge = MergeBlock(
            dim_in=dim, dim_out=dim * 2) if not last_stage else Identity()

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.merge(x)
        return x


class CSwinTransformer(nn.Layer):
    """CSwin Transformer class
    Args:
        image_size: int, input image size, default: 224
        patch_stride: int, stride for patch embedding, default: 4
        in_channels: int, num of channels of input image, default: 3
        num_classes: int, num of classes, default: 1000
        embed_dim: int, embedding dim (patch embed out dim), default: 96
        depths: list/tuple(int), number of blocks in each stage, default: [2, 4, 32, 2]
        splits: list/tuple(int), the split number in each stage, default: [1, 2, 7, 7]
        num_heads: list/tuple(int), num of attention heads in each stage, default: [4, 8, 16, 32]
        mlp_ratio: float, mlp ratio, mlp_hidden_dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if set True, qkv projection will have bias, default: True
        qk_scale: float, if set, replace the orig qk_scale (dim_head ** -0.5), default: None
        dropout: float, dropout rate for linear projection, default: 0
        attention_dropout: float, dropout rate for attention, default: 0
        droppath: float, drop path rate, default: 0
    """

    def __init__(self,
                 image_size=224,
                 patch_stride=4,
                 in_channels=3,
                 class_num=1000,
                 embed_dim=96,
                 depths=[2, 4, 32, 2],
                 splits=[1, 2, 7, 7],
                 num_heads=[4, 8, 16, 32],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        # token embedding
        self.patch_embedding = PatchEmbedding(
            patch_stride=patch_stride,
            in_channels=in_channels,
            embed_dim=embed_dim)
        # drop path decay by stage
        depth_decay = [
            x.item() for x in paddle.linspace(0, droppath, sum(depths))
        ]
        dim = embed_dim
        resolution = image_size // 4
        self.stages = nn.LayerList()
        num_stages = len(depths)
        # construct CSwin stages: each stage has multiple blocks
        for stage_idx in range(num_stages):
            stage = CSwinStage(
                dim=dim,
                input_resolution=resolution,
                depth=depths[stage_idx],
                num_heads=num_heads[stage_idx],
                split_size=splits[stage_idx],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                dropout=dropout,
                attention_dropout=attention_dropout,
                droppath=depth_decay[sum(depths[:stage_idx]):sum(
                    depths[:stage_idx + 1])],
                last_stage=stage_idx == num_stages - 1)
            self.stages.append(stage)
            if stage_idx != num_stages - 1:
                dim = dim * 2
                resolution = resolution // 2
        # last norm and classification head layers
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, class_num)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x = self.patch_embedding(x)
        for stage in self.stages:
            x = stage(x)
        x = self.norm(x)
        return paddle.mean(x, axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
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


def CSWinTransformer_tiny_224(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=224,
        embed_dim=64,
        depths=[1, 2, 21, 1],
        splits=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        droppath=0.2,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_tiny_224"],
        use_ssld=use_ssld)
    return model


def CSWinTransformer_small_224(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=224,
        embed_dim=64,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        droppath=0.4,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_small_224"],
        use_ssld=use_ssld)
    return model


def CSWinTransformer_base_224(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=224,
        embed_dim=96,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 7, 7],
        num_heads=[4, 8, 16, 32],
        droppath=0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_base_224"],
        use_ssld=use_ssld)
    return model


def CSWinTransformer_base_384(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=384,
        embed_dim=96,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 12, 12],
        num_heads=[4, 8, 16, 32],
        droppath=0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_base_384"],
        use_ssld=use_ssld)
    return model


def CSWinTransformer_large_224(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=224,
        embed_dim=144,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 7, 7],
        num_heads=[6, 12, 24, 24],
        droppath=0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_large_224"],
        use_ssld=use_ssld)
    return model


def CSWinTransformer_large_384(pretrained=False, use_ssld=False, **kwargs):
    model = CSwinTransformer(
        image_size=384,
        embed_dim=144,
        depths=[2, 4, 32, 2],
        splits=[1, 2, 12, 12],
        num_heads=[6, 12, 24, 24],
        droppath=0.5,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CSWinTransformer_large_384"],
        use_ssld=use_ssld)
    return model
