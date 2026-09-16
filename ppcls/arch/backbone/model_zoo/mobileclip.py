# -*- coding: utf-8 -*-
from .mobileclip.clip import CLIP
from ..base.registry import BACKBONES

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
