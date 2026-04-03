# -*- coding: utf-8 -*-
import os
import json
from typing import Optional, Union, Tuple, Any

import paddle
import paddle.nn as nn
from paddle.vision.transforms import (
    CenterCrop,
    Compose,
    Resize,
    ToTensor,
)

from mobileclip.clip import CLIP
from mobileclip.modules.text.tokenizer import (
    ClipTokenizer,
)
from mobileclip.modules.common.mobileone import reparameterize_model

# --- PaddleClas Registration ---
try:
    from ppcls.arch.backbone.base.registry import BACKBONES

    @BACKBONES.register
    def MobileCLIP2_S0(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_s0")
        return model

    @BACKBONES.register
    def MobileCLIP2_S2(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_s2")
        return model

    @BACKBONES.register
    def MobileCLIP2_S3(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_s3")
        return model

    @BACKBONES.register
    def MobileCLIP2_S4(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_s4")
        return model

    @BACKBONES.register
    def MobileCLIP2_B(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_b")
        return model

    @BACKBONES.register
    def MobileCLIP2_L14(pretrained=False, use_ssld=False, **kwargs):
        model = CLIP.from_variant("mobileclip2_l14")
        return model

except ImportError:
    # If not in PaddleClas environment, skipping registration
    pass

def create_model_and_transforms(
    model_name: str,
    pretrained: Optional[str] = None,
    reparameterize: Optional[bool] = True,
    device: str = "cpu",
) -> Tuple[nn.Layer, Any, Any]:
    """
    Method to instantiate model and pre-processing transforms necessary for inference.
    """
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    if not os.path.exists(model_cfg_file):
        # Check for mobileclip2 configs as well
        configs_dir = os.path.join(os.path.dirname(root_dir), "mobileclip2", "model_configs")
        model_cfg_file = os.path.join(configs_dir, model_name.upper().replace("_", "-") + ".json")
        if not os.path.exists(model_cfg_file):
            raise ValueError(f"Unsupported model name: {model_name}")
            
    with open(model_cfg_file, "r") as f:
        model_cfg = json.load(f)

    # Build preprocessing transforms for inference
    resolution = model_cfg["vision_cfg"]["image_size"] if "vision_cfg" in model_cfg else model_cfg["image_cfg"]["image_size"]
    resize_size = resolution
    centercrop_size = resolution
    aug_list = [
        Resize(
            resize_size,
            interpolation="bilinear",
        ),
        CenterCrop(centercrop_size),
        ToTensor(),
    ]
    preprocess = Compose(aug_list)

    # Build model
    # Adapt MobileCLIP2 config to CLIP class
    if "vision_cfg" in model_cfg:
        paddle_cfg = {
            "embed_dim": model_cfg["embed_dim"],
            "image_cfg": {
                "model_name": "mci0", # Assume mci0 for S0
                "image_size": model_cfg["vision_cfg"]["image_size"]
            },
            "text_cfg": {
                "context_length": model_cfg["text_cfg"]["context_length"],
                "vocab_size": model_cfg["text_cfg"]["vocab_size"],
                "dim": model_cfg["text_cfg"]["width"],
                "n_heads_per_layer": model_cfg["text_cfg"]["heads"],
                "n_transformer_layers": model_cfg["text_cfg"]["layers"],
                "ffn_multiplier_per_layer": 4.0,
                "norm_layer": "layer_norm_fp32",
                "causal_masking": not model_cfg["text_cfg"].get("no_causal_mask", True),
                "model_name": "base"
            }
        }
    else:
        paddle_cfg = model_cfg

    model = CLIP(cfg=paddle_cfg)
    paddle.set_device(device)
    model.eval()

    # Load checkpoint
    if pretrained is not None:
        chkpt = paddle.load(pretrained)
        model.set_state_dict(chkpt)

    # Reparameterize model for inference (if specified)
    if reparameterize:
        model = reparameterize_model(model)

    return model, None, preprocess


def get_tokenizer(model_name: str) -> nn.Layer:
    # Config files
    root_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(root_dir, "configs")
    model_cfg_file = os.path.join(configs_dir, model_name + ".json")

    # Get config from yaml file
    with open(model_cfg_file, "r") as f:
        model_cfg = json.load(f)

    # Build tokenizer
    text_tokenizer = ClipTokenizer(model_cfg)
    return text_tokenizer
