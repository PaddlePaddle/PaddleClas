# -*- coding: utf-8 -*-
import math
from typing import Optional, Sequence

import paddle
from paddle import Tensor, nn

from mobileclip.modules.common.transformer import (
    PositionalEmbedding,
    TransformerEncoder,
    get_normalization_layer,
)
from mobileclip.modules.text.repmixer import RepMixerBlock
from mobileclip import logger


class TextTransformer(nn.Layer):
    def __init__(self, cfg: dict, projection_dim: int, *args, **kwargs) -> None:
        super().__init__()

        model_dim = cfg["dim"]
        no_scale_embedding = cfg.get("no_scale_embedding", False)
        no_pos_embedding = cfg.get("no_pos_embedding", False)
        embed_dropout = cfg.get("embed_dropout", 0.0)
        norm_layer = cfg["norm_layer"]
        variant = cfg["model_name"]
        self.vocab_size = cfg["vocab_size"]
        self.projection_dim = projection_dim

        # Token embedding layer
        self.embedding_layer = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=model_dim
        )
        self.embed_scale = 1.0 if no_scale_embedding else model_dim**-0.5

        # Context length
        context_length = cfg["context_length"]
        assert (
            context_length is not None
        ), "Context length can't be None. Please set value accordingly."

        self.positional_embedding = (
            None
            if no_pos_embedding
            else PositionalEmbedding(
                num_embeddings=context_length, embedding_dim=model_dim
            )
        )

        self.embedding_dropout = nn.Dropout(p=embed_dropout)

        # Transformer layer
        n_transformer_layers = cfg["n_transformer_layers"]

        # FFN multipliers for transformer layer
        ffn_multipliers = cfg["ffn_multiplier_per_layer"]
        if isinstance(ffn_multipliers, (float, int)):
            ffn_multipliers = [ffn_multipliers] * n_transformer_layers

        ffn_dims = [
            int(math.ceil(model_dim * ffn_mult / 16.0) * 16.0)
            for ffn_mult in ffn_multipliers
        ]

        # Heads for transformer layers
        mha_heads = cfg["n_heads_per_layer"]
        if isinstance(mha_heads, int):
            mha_heads = [mha_heads] * n_transformer_layers

        if variant == "base":
            self.transformer = nn.LayerList(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
        elif variant == "mct":
            layers = [RepMixerBlock(dim=model_dim)]
            layers.extend(
                [
                    TransformerEncoder(
                        embed_dim=model_dim,
                        num_heads=mha_heads[layer_idx],
                        ffn_latent_dim=ffn_dims[layer_idx],
                        transformer_norm_layer=norm_layer,
                    )
                    for layer_idx in range(n_transformer_layers)
                ]
            )
            layers.append(RepMixerBlock(dim=model_dim))
            self.transformer = nn.LayerList(layers)
        else:
            raise ValueError("Unrecognized text encoder variant {}".format(variant))

        self.final_layer_norm = get_normalization_layer(
            num_features=model_dim, norm_type=norm_layer
        )

        self.projection_layer = self.create_parameter(
            shape=[model_dim, self.projection_dim],
            default_initializer=nn.initializer.XavierUniform()
        )
        self.model_dim = model_dim
        self.causal_masking = cfg["causal_masking"]

    def forward_embedding(self, text_tokens: Tensor) -> Tensor:
        token_emb = self.embedding_layer(text_tokens)
        seq_len = token_emb.shape[1]
        if self.positional_embedding is not None:
            token_emb = token_emb + self.positional_embedding(seq_len).astype(token_emb.dtype)
        token_emb = self.embedding_dropout(token_emb)
        return token_emb

    def build_attention_mask(self, context_length: int, batch_size: int) -> Tensor:
        mask = paddle.full([context_length, context_length], float("-inf"))
        mask = paddle.triu(mask, diagonal=1)
        mask = mask.unsqueeze(0).expand([batch_size, -1, -1])
        return mask

    def encode_text(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        token_emb = self.forward_embedding(text_tokens)

        attn_mask = None
        if self.causal_masking:
            attn_mask = self.build_attention_mask(
                context_length=text_tokens.shape[1], batch_size=text_tokens.shape[0]
            )
            attn_mask = attn_mask.astype(token_emb.dtype)
            key_padding_mask = None

        for layer in self.transformer:
            token_emb = layer(
                token_emb,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        token_emb = self.final_layer_norm(token_emb)

        if return_all_tokens:
            return token_emb

        # Take features from the eot embedding
        batch_indices = paddle.arange(text_tokens.shape[0])
        eot_indices = text_tokens.argmax(axis=-1)
        token_emb = paddle.stack([token_emb[i, eot_indices[i]] for i in range(text_tokens.shape[0])])

        token_emb = paddle.matmul(token_emb, self.projection_layer)
        return token_emb

    def forward(
        self,
        text_tokens: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_all_tokens: bool = False,
        *args,
        **kwargs
    ) -> Tensor:
        text_tokens = self.encode_text(
            text_tokens=text_tokens,
            key_padding_mask=key_padding_mask,
            return_all_tokens=return_all_tokens,
            *args,
            **kwargs
        )
        return text_tokens
