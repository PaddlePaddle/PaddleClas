# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Source-Code-License-NC
# PaddlePaddle port of NVLabs MambaVision.

from .mambavision_impl.mamba_vision import build_mambavision

MODEL_DOWNLOAD_URL = "https://aistudio.baidu.com/modelsdetail/51615"

_VARIANTS = {
    "MambaVision_T": "mamba_vision_T",
    "MambaVision_T2": "mamba_vision_T2",
    "MambaVision_S": "mamba_vision_S",
    "MambaVision_B": "mamba_vision_B",
    "MambaVision_B_21K": "mamba_vision_B_21k",
    "MambaVision_L": "mamba_vision_L",
    "MambaVision_L_21K": "mamba_vision_L_21k",
    "MambaVision_L2": "mamba_vision_L2",
    "MambaVision_L2_512_21K": "mamba_vision_L2_512_21k",
    "MambaVision_L3_256_21K": "mamba_vision_L3_256_21k",
    "MambaVision_L3_512_21K": "mamba_vision_L3_512_21k",
}

__all__ = list(_VARIANTS)


def _build(variant, pretrained=False, use_ssld=False, **kwargs):
    del use_ssld
    class_num = kwargs.pop("class_num", 1000)
    if pretrained is True:
        raise ValueError(
            "Automatic pretrained weight download is not available. "
            f"Download the PaddlePaddle checkpoint from {MODEL_DOWNLOAD_URL} "
            "and pass its local .pdparams path through pretrained."
        )
    checkpoint = pretrained if isinstance(pretrained, str) else None
    return build_mambavision(
        _VARIANTS[variant],
        pretrained=checkpoint,
        num_classes=class_num,
        **kwargs,
    )


def MambaVision_T(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_T", pretrained, use_ssld, **kwargs)


def MambaVision_T2(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_T2", pretrained, use_ssld, **kwargs)


def MambaVision_S(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_S", pretrained, use_ssld, **kwargs)


def MambaVision_B(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_B", pretrained, use_ssld, **kwargs)


def MambaVision_B_21K(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_B_21K", pretrained, use_ssld, **kwargs)


def MambaVision_L(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L", pretrained, use_ssld, **kwargs)


def MambaVision_L_21K(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L_21K", pretrained, use_ssld, **kwargs)


def MambaVision_L2(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L2", pretrained, use_ssld, **kwargs)


def MambaVision_L2_512_21K(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L2_512_21K", pretrained, use_ssld, **kwargs)


def MambaVision_L3_256_21K(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L3_256_21K", pretrained, use_ssld, **kwargs)


def MambaVision_L3_512_21K(pretrained=False, use_ssld=False, **kwargs):
    return _build("MambaVision_L3_512_21K", pretrained, use_ssld, **kwargs)
