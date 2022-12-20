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

from .vision_transformer import Identity
from .vision_transformer import PatchEmbed

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


class DropPath(nn.Layer):
    """DropPath class"""
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def drop_path(self, inputs):
        """drop path op
        Args:
            input: tensor with arbitrary shape
            drop_prob: float number of drop path probability, default: 0.0
            training: bool, if current mode is training, default: False
        Returns:
            output: output tensor after drop path
        """
        # if prob is 0 or eval mode, return original input
        if self.drop_prob == 0. or not self.training:
            return inputs
        keep_prob = 1 - self.drop_prob
        keep_prob = paddle.to_tensor(keep_prob, dtype='float32')
        shape = (inputs.shape[0], ) + (1, ) * (inputs.ndim - 1)  # shape=(N, 1, 1, 1)
        random_tensor = keep_prob + paddle.rand(shape, dtype=inputs.dtype)
        random_tensor = random_tensor.floor() # mask
        output = inputs.divide(keep_prob) * random_tensor # divide to keep same output expectation
        return output

    def forward(self, inputs):
        return self.drop_path(inputs)


class TalkingHeadAttention(nn.Layer):
    """ Talking head attention

    Talking head attention (https://arxiv.org/abs/2003.02436),
    applies linear projections across the attention-heads dimension,
    before and after the softmax operation.

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.):
        super(TalkingHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** (-0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(axis=-1)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)

        # talking head
        self.proj_l = nn.Linear(num_heads, num_heads)
        self.proj_w = nn.Linear(num_heads, num_heads)

    def transpose_multihead(self, x):
        new_shape = x.shape[:-1] + [self.num_heads, self.dim_head]
        x = paddle.reshape(x, new_shape)
        x = paddle.transpose(x, [0, 2, 1, 3])
        return x

    def forward(self, x):
        B, L, C = x.shape  # L: num_patches (h*w)
        qkv = self.qkv(x).chunk(3, axis=-1)
        q, k, v = map(self.transpose_multihead, qkv)  #[B, num_heads, num_patches, single_head_dim]

        q = q * self.scale
        attn = paddle.matmul(q, k, transpose_y=True)  #[B, num_heads, num_patches, num_patches]

        # projection across heads (before softmax)
        attn = paddle.transpose(attn, [0, 2, 3, 1])  #[B, num_patches, num_patches, num_heads]
        attn = self.proj_l(attn)
        attn = paddle.transpose(attn, [0, 3, 1, 2])  #[B, num_heads, num_patches, num_patches]

        attn = self.softmax(attn)

        # projection across heads (after softmax)
        attn = paddle.transpose(attn, [0, 2, 3, 1])  #[B, num_patches, num_patches, num_heads]
        attn = self.proj_w(attn)
        attn = paddle.transpose(attn, [0, 3, 1, 2])  #[B, num_heads, num_patches, num_patches]

        attn = self.attn_dropout(attn)

        z = paddle.matmul(attn, v)  #[B, num_heads, num_patches, single_head_dim]
        z = paddle.transpose(z, [0, 2, 1, 3])  #[B, num_patches, num_heads, single_head_dim]

        z = paddle.reshape(z, [B, L, C])
        z = self.proj(z)
        z = self.proj_dropout(z)

        return z

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
    def __init__(self,
                 in_features,
                 hidden_features,
                 dropout=0.):
        super(Mlp, self).__init__()
        w_attr_1, b_attr_1 = self._init_weights()
        self.fc1 = nn.Linear(in_features,
                             hidden_features,
                             weight_attr=w_attr_1,
                             bias_attr=b_attr_1)
        w_attr_2, b_attr_2 = self._init_weights()
        self.fc2 = nn.Linear(hidden_features,
                             in_features,
                             weight_attr=w_attr_2,
                             bias_attr=b_attr_2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def _init_weights(self):
        w_attr = paddle.ParamAttr(initializer=nn.initializer.TruncatedNormal(std=.02))
        b_attr = paddle.ParamAttr(initializer=nn.initializer.Constant(0.0))
        return w_attr, b_attr

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x


class ClassAttention(nn.Layer):
    """ Class Attention

    Class Attention module

    Args:
        dim: int, all heads dimension
        dim_head: int, single heads dimension, default: None
        num_heads: int, num of heads
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        qk_scale: float, if None, qk_scale is dim_head ** -0.5, default: None
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        dropout: float, dropout rate for projection dropout, default: 0.
    """
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_dropout=0.,
                 dropout=0.):
        super(ClassAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = qk_scale or self.dim_head ** (-0.5)

        self.q = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.k = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.v = nn.Linear(dim, dim, bias_attr=qkv_bias)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def forward(self, x):
        B, N, C = x.shape

        q = self.q(x[:, :1, :])  # same as x[:, 0], but more intuitive
        q = paddle.reshape(q, [B, 1, self.num_heads, self.dim_head])
        q = paddle.transpose(q, [0, 2, 1, 3])

        k = self.k(x)
        k = paddle.reshape(k, [B, N, self.num_heads, self.dim_head])
        k = paddle.transpose(k, [0, 2, 1, 3])

        v = self.v(x)
        v = paddle.reshape(v, [B, N, self.num_heads, self.dim_head])
        v = paddle.transpose(v, [0, 2, 1, 3])

        attn = paddle.matmul(q * self.scale, k, transpose_y=True)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)

        cls_embed = paddle.matmul(attn, v)
        cls_embed = paddle.transpose(cls_embed, [0, 2, 1, 3])
        cls_embed = paddle.reshape(cls_embed, [B, -1, C])
        cls_embed = self.proj(cls_embed)
        cls_embed = self.proj_dropout(cls_embed)

        return cls_embed


class LayerScaleBlock(nn.Layer):
    """ LayerScale layers

    LayerScale layers contains regular self-attention layers,
    in addition with gamma_1 and gamma_2, which apply per-channel multiplication
    after each residual block (attention and mlp layers).

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        init_values: initial values for learnable weights gamma_1 and gamma_2, default: 1e-4
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super(LayerScaleBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = TalkingHeadAttention(dim,
                                         num_heads=num_heads,
                                         qkv_bias=qkv_bias,
                                         dropout=dropout,
                                         attention_dropout=attention_dropout)
        self.drop_out = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values)
        )

        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x):
        h = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.gamma_1 * x  #[B, num_patches, embed_dim]
        x = self.drop_out(x)
        x = h + x

        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.gamma_2 * x  #[B, num_patches, embed_dim]
        x = self.drop_out(x)
        x = h + x

        return x

class LayerScaleBlockClassAttention(nn.Layer):
    """ LayerScale layers for class attention

    LayerScale layers for class attention contains regular class-attention layers,
    in addition with gamma_1 and gamma_2, which apply per-channel multiplication
    after each residual block (attention and mlp layers).

    Args:
        dim: int, all heads dimension
        num_heads: int, num of heads
        mlp_ratio: ratio to multiply on mlp input dim as mlp hidden dim, default: 4.
        qkv_bias: bool, if True, qkv linear layer is using bias, default: False
        dropout: float, dropout rate for projection dropout, default: 0.
        attention_dropout: float, dropout rate for attention dropout, default: 0.
        init_values: initial values for learnable weights gamma_1 and gamma_2, default: 1e-4
    """
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4):
        super(LayerScaleBlockClassAttention, self).__init__()
        self.norm1 = nn.LayerNorm(dim, epsilon=1e-6)
        self.attn = ClassAttention(dim=dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attention_dropout=attention_dropout,
                                   dropout=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.norm2 = nn.LayerNorm(dim, epsilon=1e-6)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       dropout=dropout)

        self.gamma_1 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values)
        )

        self.gamma_2 = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(init_values)
        )

    def forward(self, x, x_cls):
        u = paddle.concat([x_cls, x], axis=1)
        u = self.norm1(u)
        u = self.attn(u)
        u = self.gamma_1 * u
        u = self.drop_path(u)
        x_cls = u + x_cls

        h = x_cls
        x_cls = self.norm2(x_cls)
        x_cls = self.mlp(x_cls)
        x_cls = self.gamma_2 * x_cls
        x_cls = self.drop_path(x_cls)
        x_cls = h + x_cls

        return x_cls


class CaiT(nn.Layer):
    """ CaiT model
    Args:
        image_size: int, input image size, default: 224
        in_channels: int, input image channels, default: 3
        num_classes: int, num of classes, default: 1000
        patch_size: int, patch size for patch embedding, default: 16
        embed_dim: int, dim of each patch after patch embedding, default: 768
        depth: int, num of self-attention blocks, default: 12
        num_heads: int, num of attention heads, default: 12
        mlp_ratio: float, mlp hidden dim = mlp_ratio * mlp_in_dim, default: 4.
        qkv_bias: bool, if True, qkv projection is set with bias, default: True
        dropout: float, dropout rate for linear projections, default: 0.
        attention_dropout: float, dropout rate for attention, default: 0.
        droppath: float, drop path rate, default: 0.
        init_values: initial value for layer scales, default: 1e-4
        mlp_ratio_class_token: float, mlp_ratio for mlp used in class attention blocks, default: 4.0
        depth_token_only, int, num of class attention blocks, default: 2
    """
    def __init__(self,
                 image_size=224,
                 in_channels=3,
                 class_num=1000,
                 patch_size=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 dropout=0.,
                 attention_dropout=0.,
                 droppath=0.,
                 init_values=1e-4,
                 mlp_ratio_class_token=4.0,
                 depth_token_only=2):
        super().__init__()
        self.num_classes = class_num
        self.patch_embed = PatchEmbed(img_size=image_size,
                                      patch_size=patch_size,
                                      in_chans=in_channels,
                                      embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        # tokens add for classification
        self.cls_token = paddle.create_parameter(
            shape=[1, 1, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0)
        )
        # positional embeddings for patch positions
        self.pos_embed = paddle.create_parameter(
            shape=[1, num_patches, embed_dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(0.0)
        )

        self.pos_drop = nn.Dropout(dropout)

        # create self-attention(layer-scale) layers
        layer_list = []
        for i in range(depth):
            layer_list.append(LayerScaleBlock(dim=embed_dim,
                                              num_heads=num_heads,
                                              mlp_ratio=mlp_ratio,
                                              qkv_bias=qkv_bias,
                                              dropout=dropout,
                                              attention_dropout=attention_dropout,
                                              droppath=droppath,
                                              init_values=init_values))
        self.blocks = nn.LayerList(layer_list)

        # craete class-attention layers
        layer_list = []
        for i in range(depth_token_only):
            layer_list.append(LayerScaleBlockClassAttention(dim=embed_dim,
                                                            num_heads=num_heads,
                                                            mlp_ratio=mlp_ratio_class_token,
                                                            qkv_bias=qkv_bias,
                                                            dropout=dropout,
                                                            attention_dropout=attention_dropout,
                                                            droppath=droppath,
                                                            init_values=init_values))
        self.blocks_token_only = nn.LayerList(layer_list)

        self.norm = nn.LayerNorm(embed_dim, epsilon=1e-6)
        self.head = nn.Linear(embed_dim, class_num) if class_num > 0. else Identity()

    def forward_features(self, x):
        # Patch Embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand([x.shape[0], -1, -1])  # [B, 1, embed_dim]
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # Self-Attention blocks
        for idx, block in enumerate(self.blocks):
            x = block(x)  # [B, num_patches, embed_dim]
        # Class-Attention blocks
        for idx, block_token_only in enumerate(self.blocks_token_only):
            cls_tokens = block_token_only(x, cls_tokens)
        # Concat outputs
        x = paddle.concat([cls_tokens, x], axis=1)
        x = self.norm(x)  # [B, num_patches + 1, embed_dim]
        return x[:, 0]

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
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_M48'],
                     use_ssld=use_ssld)
    return model


def CaiT_M36(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_M36'],
                     use_ssld=use_ssld)
    return model


def CaiT_S36(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S36'],
                     use_ssld=use_ssld)
    return model


def CaiT_S24(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S24'],
                     use_ssld=use_ssld)
    return model


def CaiT_S24_224(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_S24_224'],
                     use_ssld=use_ssld)
    return model


def CaiT_XS24(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XS24'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS36(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS36'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS36_224(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS36_224'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS24(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS24'],
                     use_ssld=use_ssld)
    return model


def CaiT_XXS24_224(pretrained=False, use_ssld=False, **kwargs):
    model = CaiT(**kwargs)
    _load_pretrained(pretrained=pretrained,
                     model=model,
                     model_url=MODEL_URLS['CaiT_XXS24_224'],
                     use_ssld=use_ssld)
    return model
