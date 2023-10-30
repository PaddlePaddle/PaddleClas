import os
import wget
import paddle
from .model_CLIP import CLIP, LaCLIP
from paddle.vision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from .unicom import load_model


def clip_rn50():
    model = CLIP(
        embed_dim=1024,
        image_resolution=224,
        vision_layers=(3, 4, 6, 3),
        vision_width=64,
        vision_patch_size=None,
        vision_head=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    return model


def clip_vit_b_32():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_head=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    return model


def clip_vit_b_16():
    model = CLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        vision_head=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    return model


def Laclip_vit_b_32():
    model = LaCLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=32,
        vision_head=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    return model


def Laclip_vit_b_16():
    model = LaCLIP(
        embed_dim=512,
        image_resolution=224,
        vision_layers=12,
        vision_width=768,
        vision_patch_size=16,
        vision_head=64,
        context_length=77,
        vocab_size=49408,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12)
    return model


def unicom_vit_b_32():
    model = load_model("ViT-B/32")
    return model


def Laclip_vit_b_16():
    model = load_model("ViT-B/16")
    return model
