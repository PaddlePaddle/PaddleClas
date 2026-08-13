# -*- coding: utf-8 -*-
import math
from typing import Any, Optional, Dict

import paddle
import paddle.nn.functional as F
from paddle import nn

from mobileclip.text_encoder import TextTransformer
from .image_encoder import MCi

import os
import json

class CLIP(nn.Layer):
    @classmethod
    def from_variant(cls, variant_name: str, **kwargs):
        """
        Create a CLIP model from a variant name (e.g., 'mobileclip2_s0', 'mobileclip2_l14')
        """
        # 1. 映射 variant_name 到配置文件名
        config_mapping = {
            "mobileclip2_s0": "MobileCLIP2-S0.json",
            "mobileclip2_s2": "MobileCLIP2-S2.json",
            "mobileclip2_s3": "MobileCLIP2-S3.json",
            "mobileclip2_s4": "MobileCLIP2-S4.json",
            "mobileclip2_b": "MobileCLIP2-B.json",
            "mobileclip2_l14": "MobileCLIP2-L-14.json",
            # 可以根据需要添加更多映射...
        }
        
        config_file = config_mapping.get(variant_name.lower())
        if not config_file:
            raise ValueError(f"Unknown variant name: {variant_name}. Available: {list(config_mapping.keys())}")
            
        # 2. 读取配置文件
        config_path = os.path.join(os.path.dirname(__file__), "configs", config_file)
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            cfg = json.load(f)
            
        # 3. 转换为 CLIP 需要的 paddle_cfg 格式
        image_model_name = "mci0"
        if "s2" in variant_name.lower(): image_model_name = "mci1"
        elif "s3" in variant_name.lower(): image_model_name = "mci2"
        elif "s4" in variant_name.lower(): image_model_name = "mci2" # S4 也是 mci2
        elif "b" in variant_name.lower(): image_model_name = "vit_b16"
        elif "l14" in variant_name.lower(): image_model_name = "vit_l14"
        
        paddle_cfg = {
            "embed_dim": cfg["embed_dim"],
            "image_cfg": {
                "model_name": image_model_name,
                "image_size": cfg["vision_cfg"]["image_size"]
            },
            "text_cfg": {
                "context_length": cfg["text_cfg"]["context_length"],
                "vocab_size": cfg["text_cfg"]["vocab_size"],
                "dim": cfg["text_cfg"]["width"],
                "n_heads_per_layer": cfg["text_cfg"]["heads"],
                "n_transformer_layers": cfg["text_cfg"]["layers"],
                "ffn_multiplier_per_layer": 4.0,
                "norm_layer": "layer_norm_fp32",
                "causal_masking": not cfg["text_cfg"].get("no_causal_mask", True),
                "model_name": "base"
            }
        }
        
        return cls(cfg=paddle_cfg, **kwargs)

    def __init__(self, cfg: Dict, output_dict: bool = False, *args, **kwargs) -> None:
        super().__init__()
        self.output_dict = output_dict
        self.projection_dim = cfg["embed_dim"]
        if self.projection_dim is None:
            raise ValueError("Please specify `embed_dim` in model config.")

        self.image_encoder = MCi(
            model_name=cfg["image_cfg"]["model_name"],
            projection_dim=self.projection_dim,
        )
        self.text_encoder = TextTransformer(
            cfg=cfg["text_cfg"], projection_dim=self.projection_dim
        )
        self.logit_scale = self.create_parameter(
            shape=[],
            default_initializer=nn.initializer.Constant(math.log(1.0 / 0.07))
        )

    def _exponentiate_and_clip_logits(self, max_scale: float = 100.0):
        scale = paddle.exp(self.logit_scale)
        scale = paddle.clip(scale, 0, max_scale)
        return scale

    def encode_image(self, image: paddle.Tensor, normalize: bool = False):
        image_encoder_out = self.image_encoder(image)
        if isinstance(image_encoder_out, dict):
            features = image_encoder_out["logits"]
        else:
            features = image_encoder_out
        return F.normalize(features, axis=-1) if normalize else features

    def encode_text(self, text: paddle.Tensor, normalize: bool = False):
        text_features = self.text_encoder(text_tokens=text, key_padding_mask=None)
        return F.normalize(text_features, axis=-1) if normalize else text_features

    def forward(
        self,
        image: Optional[paddle.Tensor] = None,
        text: Optional[paddle.Tensor] = None,
        *args,
        **kwargs
    ) -> Any:
        image_embeddings = (
            self.encode_image(image, normalize=True) if image is not None else None
        )
        text_embeddings = (
            self.encode_text(text, normalize=True) if text is not None else None
        )

        if self.output_dict:
            return {
                "image_features": image_embeddings,
                "text_features": text_embeddings,
                "logit_scale": self._exponentiate_and_clip_logits(),
            }
        return image_embeddings, text_embeddings, self._exponentiate_and_clip_logits()
