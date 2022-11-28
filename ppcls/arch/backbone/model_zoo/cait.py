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

# Code was heavily based on https://github.com/facebookresearch/deit
# reference: https://arxiv.org/abs/2103.17239

import paddle
import paddle.nn as nn
from .vision_transformer import VisionTransformer, Identity, DropPath, Mlp, PatchEmbed

from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {
    "CaiT_M48": "",
    "CaiT_M36": "",
    "CaiT_S36": "",
    "CaiT_S24": "",
    "CaiT_S24_224": "",
    "CaiT_XS24": "",
    "CaiT_XXS24": "",
    "CaiT_XXS24_224": "",
    "CaiT_XXS36": "",
    "CaiT_XXS36_224": "",
}

__all__ = list(MODEL_URLS.keys())


class ClassAttention(nn.Layer):
    """
    Class Attention module
    """
    def __init__(self, dim, num_heads=8,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x[:,0]).unsqueeze(1)
        q = q.reshape([B, 1, self.num_heads, C // self.num_heads])
        q = q.transpose(0, 2, 1, 3)

        k = self.k(x)
        k = k.reshape([B, N, self.num_heads, C // self.num_heads])
        k = k.transpose([0, 2, 1, 3])

        v = self.v(x)
        v = v.reshape([B, N, self.num_heads, C // self.num_heads])
        v = v.transpose([0, 2, 1, 3])

        q = q * self.scale
        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2]))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x_cls = paddle.matmul(attn, v)
        x_cls = x_cls.transpose([0, 2, 1, 3])
        x_cls = x_cls.reshape([B, 1, C])
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)
        return x_cls


class LayerScaleBlockClassAttention(nn.Layer):
    """
    LayerScale layers for class attention
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 Attention_block=ClassAttention, Mlp_block=Mlp,
                 init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim,
                             hidden_features=mlp_hidden_dim,
                             act_layer=act_layer,
                             drop=drop)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x, x_cls):
        u = paddle.concat([x_cls, x], axis=1)

        x_cls = x_cls + self.drop_path(self.gamma_1 * self.attn(self.norm1(u)))

        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))

        return x_cls


class AttentionTalkingHead(nn.Layer):
    """
    Talking Heads Attention
    """
    def __init__(self, dim, num_heads=8,
                 qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)

        # talking head
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        new_shape = tuple(x.shape[:-1]) + (self.num_heads, self.dim_head)
        x = x.reshape(new_shape)
        x = x.transpose([0, 2, 1, 3])
        return x

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape([B, N, 3, self.num_heads, C // self.num_heads])
        qkv = qkv.transpose([2, 0, 3, 1, 4])
        q, k, v = qkv[0] * self.scale , qkv[1], qkv[2] 

        attn = paddle.matmul(q, k.transpose([0, 1, 3, 2]))

        # projection across heads (before softmax)
        attn = attn.transpose([0, 2, 3, 1]) #[B, num_patches, num_patches, num_heads]
        attn = self.proj_l(attn)
        attn = attn.transpose([0, 3, 1, 2]) #[B, num_heads, num_patches, num_patches]

        attn = self.softmax(attn)

        # projection across heads (after softmax)
        attn = attn.transpose([0, 2, 3, 1]) #[B, num_patches, num_patches, num_heads]
        attn = self.proj_w(attn)
        attn = attn.transpose([0, 3, 1, 2]) #[B, num_heads, num_patches, num_patches]

        attn = self.attn_drop(attn)

        x = paddle.matmul(attn, v) #[B, num_heads, num_patches, single_head_dim]
        x = x.transpose([0, 2, 1, 3]) #[B, num_patches, num_heads, single_head_dim]
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScaleBlock(nn.Layer):
    """
    LayerScale layers
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 Attention_block=AttentionTalkingHead,
                 Mlp_block=Mlp, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_block(dim,
                                    num_heads=num_heads,
                                    qkv_bias=qkv_bias,
                                    qk_scale=qk_scale,
                                    attn_drop=attn_drop,
                                    proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim,
                             hidden_features=mlp_hidden_dim,
                             act_layer=act_layer,
                             drop=drop)
        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))
        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values))

    def forward(self, x):
        x = x + self.drop_path(self.gamma_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
        return x


class Cait(VisionTransformer):
    """
    CaiT Model
    """
    def __init__(self, image_size=224, patch_size=16, in_channels=3, class_num=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, global_pool=None,
                 block_layers=LayerScaleBlock,
                 block_layers_token=LayerScaleBlockClassAttention,
                 Patch_layer=PatchEmbed,
                 act_layer=nn.GELU,
                 Attention_block=AttentionTalkingHead,
                 Mlp_block=Mlp,
                 init_scale=1e-4,
                 Attention_block_token_only=ClassAttention,
                 Mlp_block_token_only=Mlp,
                 depth_token_only=2,
                 mlp_ratio_clstk = 4.0):
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        # convert image to paches
        self.patch_embed = Patch_layer(img_size=image_size,
                                       patch_size=patch_size,
                                       in_chans=in_channels,
                                       embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0))
        self.pos_drop = nn.Dropout(drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.LayerList([
            block_layers(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block, Mlp_block=Mlp_block, init_values=init_scale
            ) for i in range(depth)
        ])

        self.blocks_token_only = nn.LayerList([
            block_layers_token(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio_clstk, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=0.0, attn_drop=0.0, drop_path=0.0, norm_layer=norm_layer,
                act_layer=act_layer, Attention_block=Attention_block_token_only,
                Mlp_block=Mlp_block_token_only, init_values=init_scale
            ) for i in range(depth_token_only)
        ])

        self.norm = norm_layer(embed_dim, epsilon=1e-6)
        self.feature_info = [dict(num_chs=embed_dim, reduction=0, module='head')]
        self.head = nn.Linear(embed_dim, class_num) if class_num > 0 else Identity()

    def forward_features(self, x):
        # Patch Embedding
        B = x.shape[0]
        x = self.patch_embed(x) # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand([B, -1, -1]) # [B, 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Self-Attention blocks
        for idx, block in enumerate(self.blocks):
            x = block(x) # [B, num_patches, embed_dim]
        # Class-Attention blocks
        for idx, block in enumerate(self.blocks_token_only):
            cls_tokens = block(x, cls_tokens) # [B, 1, embed_dim]
        # Concat outputs
        x = paddle.concat([cls_tokens, x], axis=1)
        x = self.norm(x) # [B, num_patches + 1, embed_dim]
        return x[:, 0] # returns only cls_tokens

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def CaiT_M48(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=448,
                 patch_size=16,
                 embed_dim=768,
                 depth=48,
                 num_heads=16,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-6,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_M48'],
                     use_ssld=use_ssld)
    return model


def CaiT_M36(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=768,
                 depth=36,
                 num_heads=16,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-6,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_M36'],
                     use_ssld=use_ssld)
    return model


def CaiT_S36(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=384,
                 depth=36,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-6,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S36'],
                     use_ssld=use_ssld)
    return model


def CaiT_S24(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=384,
                 depth=24,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S24'],
                     use_ssld=use_ssld)
    return model


def CaiT_S24_224(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=224,
                 patch_size=16,
                 embed_dim=384,
                 depth=24,
                 num_heads=8,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S24_224'],
                     use_ssld=use_ssld)
    return model


def CaiT_XS24(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=288,
                 depth=24,
                 num_heads=6,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XS24'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS36(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=192,
                 depth=36,
                 num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS36'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS36_224(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=224,
                 patch_size=16,
                 embed_dim=192,
                 depth=36,
                 num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS36_224'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS24(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=384,
                 patch_size=16,
                 embed_dim=192,
                 depth=24,
                 num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS24'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS24_224(pretrained=False, use_ssld=False, **kwargs):
    model = Cait(image_size=224,
                 patch_size=16,
                 embed_dim=192,
                 depth=24,
                 num_heads=4,
                 mlp_ratio=4,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 init_scale=1e-5,
                 depth_token_only=2,
                 **kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS24_224'],
                     use_ssld=use_ssld)
    return model
