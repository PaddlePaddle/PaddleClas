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

# Code was based on https://github.com/Sense-X/UniFormer
# reference: https://arxiv.org/abs/2201.09450

from collections import OrderedDict
from functools import partial
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import math
from .vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity, Mlp

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "UniFormer_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/UniFormer_small_pretrained.pdparams",
    "UniFormer_small_plus":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/UniFormer_small_plus_pretrained.pdparams",
    "UniFormer_small_plus_dim64":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/UniFormer_small_plus_dim64_pretrained.pdparams",
    "UniFormer_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/UniFormer_base_pretrained.pdparams",
    "UniFormer_base_ls":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/UniFormer_base_ls_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

layer_scale = False
init_value = 1e-6


class CMlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1_conv = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2_conv = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1_conv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2_conv(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(
            shape=[B, N, 3, self.num_heads, C // self.num_heads]).transpose(
                perm=[2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @k.transpose(perm=[0, 1, 3, 2])) * self.scale
        attn = nn.Softmax(axis=-1)(attn)
        attn = self.attn_drop(attn)

        x = (attn @v).transpose(perm=[0, 2, 1, 3]).reshape(shape=[B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2D(dim)
        self.conv1 = nn.Conv2D(dim, dim, 1)
        self.conv2 = nn.Conv2D(dim, dim, 1)
        self.attn = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = x + self.drop_path(
            self.conv2(self.attn(self.conv1(self.norm1(x)))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        global layer_scale
        self.ls = layer_scale
        if self.ls:
            global init_value
            print(f"Use layer_scale: {layer_scale}, init_values: {init_value}")
            self.gamma_1 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=init_value))
            self.gamma_2 = self.create_parameter(
                [dim],
                dtype='float32',
                default_initializer=nn.initializer.Constant(value=init_value))

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, N, H, W = x.shape
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        if self.ls:
            x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
            x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(perm=[0, 2, 1]).reshape(shape=[B, N, H, W])
        return x


class HeadEmbedding(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2D(out_channels // 2),
            nn.GELU(),
            nn.Conv2D(
                out_channels // 2,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2D(out_channels))

    def forward(self, x):
        x = self.proj(x)
        return x


class MiddleEmbedding(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)),
            nn.BatchNorm2D(out_channels))

    def forward(self, x):
        x = self.proj(x)
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj_conv = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj_conv(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(perm=[0, 2, 1])
        x = self.norm(x)
        x = x.reshape(shape=[B, H, W, C]).transpose(perm=[0, 3, 1, 2])
        return x


class UniFormer(nn.Layer):
    """ UniFormer
    A PaddlePaddle impl of : `UniFormer: Unifying Convolution and Self-attention for Visual Recognition`  -
        https://arxiv.org/abs/2201.09450
    """

    def __init__(self,
                 depth=[3, 4, 8, 3],
                 img_size=224,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=[64, 128, 320, 512],
                 head_dim=64,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=None,
                 conv_stem=False):
        """
        Args:
            depth (list): depth of each stage
            img_size (int, tuple): input image size
            in_chans (int): number of input channels
            class_num (int): number of classes for classification head
            embed_dim (list): embedding dimension of each stage
            head_dim (int): head dimension
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer (nn.Module): normalization layer
            conv_stem (bool): whether use overlapped patch stem
        """
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        if conv_stem:
            self.patch_embed1 = HeadEmbedding(
                in_channels=in_chans, out_channels=embed_dim[0])
            self.patch_embed2 = MiddleEmbedding(
                in_channels=embed_dim[0], out_channels=embed_dim[1])
            self.patch_embed3 = MiddleEmbedding(
                in_channels=embed_dim[1], out_channels=embed_dim[2])
            self.patch_embed4 = MiddleEmbedding(
                in_channels=embed_dim[2], out_channels=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size,
                patch_size=4,
                in_chans=in_chans,
                embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4,
                patch_size=2,
                in_chans=embed_dim[0],
                embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8,
                patch_size=2,
                in_chans=embed_dim[1],
                embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16,
                patch_size=2,
                in_chans=embed_dim[2],
                embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depth))
        ]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.LayerList([
            CBlock(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer) for i in range(depth[0])
        ])
        self.blocks2 = nn.LayerList([
            CBlock(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0]],
                norm_layer=norm_layer) for i in range(depth[1])
        ])
        self.blocks3 = nn.LayerList([
            SABlock(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1]],
                norm_layer=norm_layer) for i in range(depth[2])
        ])
        self.blocks4 = nn.LayerList([
            SABlock(
                dim=embed_dim[3],
                num_heads=num_heads[3],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                norm_layer=norm_layer) for i in range(depth[3])
        ])
        self.norm = nn.BatchNorm2D(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(embed_dim, representation_size)),
                             ('act', nn.Tanh())]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1],
                              class_num) if class_num > 0 else nn.Identity()

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
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.flatten(2).mean(-1)
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


def UniFormer_small(pretrained=True, use_ssld=False, **kwargs):
    model = UniFormer(
        depth=[3, 4, 8, 3],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        drop_path_rate=0.1,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["UniFormer_small"], use_ssld=use_ssld)
    return model


def UniFormer_small_plus(pretrained=True, use_ssld=False, **kwargs):
    model = UniFormer(
        depth=[3, 5, 9, 3],
        conv_stem=True,
        embed_dim=[64, 128, 320, 512],
        head_dim=32,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        drop_path_rate=0.1,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["UniFormer_small_plus"],
        use_ssld=use_ssld)
    return model


def UniFormer_small_plus_dim64(pretrained=True, use_ssld=False, **kwargs):
    model = UniFormer(
        depth=[3, 5, 9, 3],
        conv_stem=True,
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        drop_path_rate=0.1,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["UniFormer_small_plus_dim64"],
        use_ssld=use_ssld)
    return model


def UniFormer_base(pretrained=True, use_ssld=False, **kwargs):
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        drop_path_rate=0.3,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["UniFormer_base"], use_ssld=use_ssld)
    return model


def UniFormer_base_ls(pretrained=True, use_ssld=False, **kwargs):
    global layer_scale
    layer_scale = True
    model = UniFormer(
        depth=[5, 8, 20, 7],
        embed_dim=[64, 128, 320, 512],
        head_dim=64,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, epsilon=1e-6),
        drop_path_rate=0.3,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["UniFormer_base_ls"], use_ssld=use_ssld)
    return model
