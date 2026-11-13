# Copyright (c) 2025, NVIDIA Corporation. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-Source-Code-License-NC
# PaddlePaddle port of NVLabs MambaVision.

import sys
from pathlib import Path

import paddle.nn as nn

import paddle

from .mamba_block import ChannelFirstLayerNorm, LayerNorm2D, MambaVisionLayer, PatchEmbed

MODEL_CONFIGS = {
    "mamba_vision_T":
    dict(
        depths=[1, 3, 8, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        layer_scale=None,
    ),
    "mamba_vision_T2":
    dict(
        depths=[1, 3, 11, 4],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=80,
        in_dim=32,
        mlp_ratio=4,
        drop_path_rate=0.2,
        layer_scale=None,
    ),
    "mamba_vision_S":
    dict(
        depths=[3, 3, 7, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=96,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.2,
        layer_scale=None,
    ),
    "mamba_vision_B":
    dict(
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_B_21k":
    dict(
        depths=[3, 3, 10, 5],
        num_heads=[2, 4, 8, 16],
        window_size=[8, 8, 14, 7],
        dim=128,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_L":
    dict(
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_L_21k":
    dict(
        depths=[3, 3, 10, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_L2":
    dict(
        depths=[3, 3, 12, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 14, 7],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_L2_512_21k":
    dict(
        depths=[3, 3, 12, 5],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 32, 16],
        dim=196,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.3,
        layer_scale=1e-5,
    ),
    "mamba_vision_L3_256_21k":
    dict(
        depths=[3, 3, 20, 10],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 16, 8],
        dim=256,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.5,
        layer_scale=1e-5,
    ),
    "mamba_vision_L3_512_21k":
    dict(
        depths=[3, 3, 20, 10],
        num_heads=[4, 8, 16, 32],
        window_size=[8, 8, 32, 16],
        dim=256,
        in_dim=64,
        mlp_ratio=4,
        drop_path_rate=0.5,
        layer_scale=1e-5,
    ),
}


def _register_with_paddleclas(cls):
    # Native PaddleClas registration is provided by ppcls.arch.backbone.__init__.
    return cls


class MambaVision(nn.Layer):

    def __init__(
        self,
        dim,
        in_dim,
        depths,
        window_size,
        mlp_ratio,
        num_heads,
        drop_path_rate=0.2,
        in_chans=3,
        num_classes=1000,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        layer_scale=None,
        layer_scale_conv=None,
        scan_impl="sequential",
        **kwargs,
    ):
        super().__init__()
        num_features = int(dim * 2**(len(depths) - 1))
        self.num_classes = num_classes
        self.num_features = num_features
        self.dims = [int(dim * 2**i) for i in range(len(depths))]
        self.patch_embed = PatchEmbed(in_chans=in_chans,
                                      in_dim=in_dim,
                                      dim=dim)
        dpr = paddle.linspace(0, drop_path_rate, sum(depths)).numpy().tolist()
        self.levels = nn.LayerList()
        for i in range(len(depths)):
            conv = i == 0 or i == 1
            if depths[i] % 2 != 0:
                transformer_blocks = list(range(depths[i] // 2 + 1, depths[i]))
            else:
                transformer_blocks = list(range(depths[i] // 2, depths[i]))
            level = MambaVisionLayer(
                dim=int(dim * 2**i),
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size[i],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                conv=conv,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=(i < 3),
                layer_scale=layer_scale,
                layer_scale_conv=layer_scale_conv,
                transformer_blocks=transformer_blocks,
                scan_impl=scan_impl,
            )
            self.levels.append(level)
        self.norm = nn.BatchNorm2D(num_features)
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        self.head = nn.Linear(
            num_features, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = paddle.flatten(x, 1)
        return x

    def forward_intermediates(self,
                              x,
                              out_indices=(0, 1, 2, 3),
                              norm_layer=None):
        x = self.patch_embed(x)
        outs = []
        for i, level in enumerate(self.levels):
            x, stage = level(x, return_stage_output=True)
            if i in out_indices:
                if norm_layer is not None:
                    stage = norm_layer[i](stage)
                outs.append(stage)
        return outs if len(out_indices) else x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@_register_with_paddleclas
class MambaVisionBackbone(MambaVision):

    def __init__(
            self,
            dim,
            in_dim,
            depths,
            window_size,
            mlp_ratio,
            num_heads,
            out_indices=(0, 1, 2, 3),
            norm_layer="ln2d",
            **kwargs,
    ):
        super().__init__(
            dim=dim,
            in_dim=in_dim,
            depths=depths,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            num_heads=num_heads,
            **kwargs,
        )
        self.out_indices = out_indices
        norm_layer = norm_layer.lower()
        norm_cls = {
            "ln2d": LayerNorm2D,
            "bn": nn.BatchNorm2D,
            "ln": ChannelFirstLayerNorm,
        }.get(norm_layer)
        if norm_cls is None and norm_layer != "ln":
            raise ValueError(f"Unsupported norm_layer: {norm_layer}")
        self.outnorms = nn.LayerList()
        for i, channels in enumerate(self.dims):
            if i in out_indices:
                self.outnorms.append(norm_cls(channels))
            else:
                self.outnorms.append(nn.Identity())
        del self.norm
        del self.head

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        for i, level in enumerate(self.levels):
            x, stage = level(x, return_stage_output=True)
            if i in self.out_indices:
                out = self.outnorms[i](stage)
                outs.append(out)
        return outs if len(self.out_indices) else x


@_register_with_paddleclas
class MambaVisionClassifier(nn.Layer):
    """PaddleClas Arch-compatible classification wrapper."""

    def __init__(
        self,
        model_name="mamba_vision_T",
        pretrained=None,
        class_num=1000,
        **kwargs,
    ):
        super().__init__()
        kwargs.setdefault("num_classes", class_num)
        self.model = build_mambavision(model_name=model_name,
                                       pretrained=pretrained,
                                       **kwargs)

    def forward(self, x):
        return self.model(x)


def build_mambavision(model_name="mamba_vision_T", pretrained=None, **kwargs):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MambaVision variant: {model_name}")
    cfg = MODEL_CONFIGS[model_name].copy()
    cfg.update(kwargs)
    model = MambaVision(**cfg)
    if pretrained:
        state = paddle.load(pretrained)
        model.set_state_dict(state)
    return model


def load_matching_mambavision_checkpoint(model, checkpoint_path):
    """Load only tensors whose names and shapes match a MambaVision model.

    This is intended for classification transfer learning, where a 1000-class
    ImageNet checkpoint must initialize a model with a different classifier head.
    The normal ``pretrained=`` path remains strict so alignment tests still catch
    incomplete or mismatched converted checkpoints.
    """
    state = paddle.load(checkpoint_path)
    current_state = model.state_dict()
    matched = {}
    skipped_missing = []
    skipped_shape = []
    for name, value in state.items():
        if name not in current_state:
            skipped_missing.append(name)
        elif list(value.shape) != list(current_state[name].shape):
            skipped_shape.append({
                "name":
                name,
                "checkpoint_shape":
                list(value.shape),
                "model_shape":
                list(current_state[name].shape),
            })
        else:
            matched[name] = value
    current_state.update(matched)
    model.set_state_dict(current_state)
    return {
        "matched_tensors": len(matched),
        "checkpoint_tensors": len(state),
        "skipped_missing": skipped_missing,
        "skipped_shape": skipped_shape,
    }


def build_mambavision_backbone(model_name="mamba_vision_T",
                               pretrained=None,
                               **kwargs):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown MambaVision variant: {model_name}")
    cfg = MODEL_CONFIGS[model_name].copy()
    cfg.update(kwargs)
    model = MambaVisionBackbone(**cfg)
    if pretrained:
        state = paddle.load(pretrained)
        current_state = model.state_dict()
        current_state.update({
            k: v
            for k, v in state.items() if k in current_state
        })
        model.set_state_dict(current_state)
    return model


def mamba_vision_T(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_T", pretrained=pretrained, **kwargs)


def mamba_vision_T2(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_T2",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_S(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_S", pretrained=pretrained, **kwargs)


def mamba_vision_B(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_B", pretrained=pretrained, **kwargs)


def mamba_vision_B_21k(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_B_21k",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_L(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L", pretrained=pretrained, **kwargs)


def mamba_vision_L_21k(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L_21k",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_L2(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L2",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_L2_512_21k(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L2_512_21k",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_L3_256_21k(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L3_256_21k",
                             pretrained=pretrained,
                             **kwargs)


def mamba_vision_L3_512_21k(pretrained=None, **kwargs):
    return build_mambavision("mamba_vision_L3_512_21k",
                             pretrained=pretrained,
                             **kwargs)
