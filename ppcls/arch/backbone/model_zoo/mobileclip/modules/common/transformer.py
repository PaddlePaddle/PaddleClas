# -*- coding: utf-8 -*-
from typing import List, Optional, Union

import paddle
from paddle import Tensor, nn
import paddle.nn.functional as F

from mobileclip import logger


class LayerNormFP32(nn.LayerNorm):
    """
    Applies `Layer Normalization <https://arxiv.org/abs/1607.06450>`_ over a input tensor with FP32 precision
    """

    def __init__(
        self,
        normalized_shape: Union[int, List[int]],
        eps: Optional[float] = 1e-5,
        elementwise_affine: Optional[bool] = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            normalized_shape=normalized_shape,
            epsilon=eps,
            weight_attr=None if elementwise_affine else False,
            bias_attr=None if elementwise_affine else False,
            *args,
            **kwargs,
        )

    def forward(self, x: Tensor) -> Tensor:
        inp_dtype = x.dtype
        return super().forward(x.astype("float32")).astype(inp_dtype)


def get_normalization_layer(norm_type, num_features):
    if norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "layer_norm_fp32":
        return LayerNormFP32(num_features)
    else:
        raise NotImplementedError(f"Option: {norm_type} not supported.")


class PositionalEmbedding(nn.Layer):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        is_learnable: Optional[bool] = False,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pos_embed = LearnablePositionalEmbedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            interpolation_mode=interpolation_mode,
            *args,
            **kwargs,
        )

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        return self.pos_embed(seq_len, *args, **kwargs)


class LearnablePositionalEmbedding(nn.Layer):
    """Learnable Positional embedding"""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        interpolation_mode: Optional[str] = "bilinear",
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pos_embed = self.create_parameter(
            shape=[1, 1, num_embeddings, embedding_dim],
            default_initializer=nn.initializer.TruncatedNormal(std=embedding_dim**-0.5)
        )
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.padding_idx = padding_idx
        self.interpolation_mode = interpolation_mode

    def forward(self, seq_len: int, *args, **kwargs) -> Tensor:
        pos_embed = self.pos_embed
        if self.padding_idx is not None:
            with paddle.no_grad():
                pos_embed[:, :, self.padding_idx, ...] = 0.0

        if seq_len != self.num_embeddings:
            pos_embed = F.interpolate(
                pos_embed,
                size=(seq_len, self.embedding_dim),
                mode=self.interpolation_mode,
            )

        return pos_embed.reshape([1, seq_len, self.embedding_dim])


class MultiHeadAttention(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        output_dim: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        if output_dim is None:
            output_dim = embed_dim
        super().__init__()
        self.qkv_proj = nn.Linear(
            in_features=embed_dim, out_features=3 * embed_dim, bias_attr=bias
        )
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.out_proj = nn.Linear(
            in_features=embed_dim, out_features=output_dim, bias_attr=bias
        )
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim**-0.5
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(
        self,
        x_q: Tensor,
        x_kv: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        b_sz, S_len, _ = x_q.shape
        qkv = self.qkv_proj(x_q).reshape([b_sz, S_len, 3, self.num_heads, -1])
        qkv = qkv.transpose([0, 3, 2, 1, 4])
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        
        # Scale query
        query = query * self.scaling
        
        # Attention scores
        attn = paddle.matmul(query, key.transpose([0, 1, 3, 2]))
        if attn_mask is not None:
            attn += attn_mask
        
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_dropout(attn)
        
        out = paddle.matmul(attn, value)
        out = out.transpose([0, 2, 1, 3]).reshape([b_sz, S_len, -1])
        out = self.out_proj(out)
        return out


class TransformerEncoder(nn.Layer):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ffn_latent_dim: int,
        transformer_norm_layer: str = "layer_norm",
        dropout: float = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pre_norm_mha = nn.Sequential(
            get_normalization_layer(transformer_norm_layer, embed_dim),
            MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, *args, **kwargs)
        )
        self.pre_norm_ffn = nn.Sequential(
            get_normalization_layer(transformer_norm_layer, embed_dim),
            nn.Linear(embed_dim, ffn_latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_latent_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = x + self.pre_norm_mha[1](self.pre_norm_mha[0](x), *args, **kwargs)
        x = x + self.pre_norm_ffn(x)
        return x
