# copyright (c) 2024 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/2303.03667

import os
import math
import copy
import warnings

import paddle
import paddle.nn as nn

from .vision_transformer import trunc_normal_, zeros_, ones_
from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import DropPath

MODEL_URLS = {
    "FasterNet_T0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_T0_pretrained.pdparams",
    "FasterNet_T1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_T1_pretrained.pdparams",
    "FasterNet_T2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_T2_pretrained.pdparams",
    "FasterNet_S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_S_pretrained.pdparams",
    "FasterNet_M":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_M_pretrained.pdparams",
    "FasterNet_L":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/FasterNet_L_pretrained.pdparams",
}

__all__ = MODEL_URLS.keys()

NET_CONFIG = {
    "FasterNet_T0":
    [3, 40, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.0, 0, 'BN', 'GELU'],
    "FasterNet_T1": [
        3, 64, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.02, 0, 'BN',
        'GELU'
    ],
    "FasterNet_T2": [
        3, 96, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.05, 0, 'BN',
        'RELU'
    ],
    "FasterNet_S": [
        3, 128, [1, 2, 13, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.1, 0, 'BN',
        'RELU'
    ],
    "FasterNet_M": [
        3, 144, [3, 4, 18, 3], 2, 4, 4, 4, 2, 2, True, 1280, 0.2, 0, 'BN',
        'RELU'
    ],
    "FasterNet_L": [
        3, 192, [3, 4, 18, 3], 2, 4, 4, 4, 2, 2, True, 1280, 0.3, 0, 'BN',
        'RELU'
    ],
}


class PartialConv(nn.Layer):
    def __init__(self, dim: int, n_div: int, forward: str):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2D(
            in_channels=self.dim_conv3,
            out_channels=self.dim_conv3,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError(
                f"Forward method '{forward}' is not implemented.")

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(
            x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = paddle.split(
            x=x, num_or_sections=[self.dim_conv3, self.dim_untouched], axis=1)
        x1 = self.partial_conv3(x1)
        x = paddle.concat(x=(x1, x2), axis=1)
        return x


class MLPBlock(nn.Layer):
    def __init__(self, dim, n_div, mlp_ratio, drop_path,
                 layer_scale_init_value, act_layer, norm_layer, pconv_fw_type):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        if drop_path > 0.:
            self.drop_path = DropPath(drop_path)
        else:
            self.drop_path = nn.Identity()
        self.n_div = n_div
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer = [
            nn.Conv2D(
                in_channels=dim,
                out_channels=mlp_hidden_dim,
                kernel_size=1,
                bias_attr=False), norm_layer(mlp_hidden_dim), act_layer(),
            nn.Conv2D(
                in_channels=mlp_hidden_dim,
                out_channels=dim,
                kernel_size=1,
                bias_attr=False)
        ]
        self.mlp = nn.Sequential(*mlp_layer)
        self.spatial_mixing = PartialConv(dim, n_div, pconv_fw_type)
        if layer_scale_init_value > 0:
            self.layer_scale = (
                paddle.base.framework.EagerParamBase.from_tensor(
                    tensor=layer_scale_init_value * paddle.ones(shape=dim),
                    trainable=True))
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x):
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(
            self.layer_scale.unsqueeze(axis=-1).unsqueeze(axis=-1) *
            self.mlp(x))
        return x


class BasicStage(nn.Layer):
    def __init__(self, dim, depth, n_div, mlp_ratio, drop_path,
                 layer_scale_init_value, norm_layer, act_layer, pconv_fw_type):
        super().__init__()
        blocks_list = [
            MLPBlock(
                dim=dim,
                n_div=n_div,
                mlp_ratio=mlp_ratio,
                drop_path=drop_path[i],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type) for i in range(depth)
        ]
        self.blocks = nn.Sequential(*blocks_list)

    def forward(self, x):
        x = self.blocks(x)
        return x


class PatchEmbed(nn.Layer):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim,
                 norm_layer):
        super().__init__()
        self.proj = nn.Conv2D(
            in_channels=in_chans,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            bias_attr=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, momentum=0.1)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.proj(x))
        return x


class PatchMerging(nn.Layer):
    def __init__(self, patch_size_t, patch_stride_t, dim, norm_layer):
        super().__init__()
        self.reduction = nn.Conv2D(
            in_channels=dim,
            out_channels=2 * dim,
            kernel_size=patch_size_t,
            stride=patch_stride_t,
            bias_attr=False)
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x):
        x = self.norm(self.reduction(x))
        return x


class FasterNet(nn.Layer):
    """
    FasterNet
    Args:
        in_chans: int=3. Number of input channels. Default value is 3.
        embed_dim: int=96. The dimension of embedding. Default value is 96.
        depths: tuple=(1, 2, 8, 2). The depth of each stage. Default value is (1, 2, 8, 2).
        mlp_ratio: float=2.0. The ratio of hidden dimension to embedding dimension. Default value is 2.0.
        n_div: int=4. The number of divisions in the spatial dimension. Default value is 4.
        patch_size: int=4. The size of patch. Default value is 4.
        patch_stride: int=4. The stride of patch. Default value is 4.
        patch_size_t: int=2. The size of patch for merging. Default value is 2.
        patch_stride_t: int=2. The stride of patch for merging. Default value is 2.
        patch_norm: bool=True. Whether to use patch normalization. Default value is True.
        feature_dim: int=1280. The dimension of feature. Default value is 1280.
        drop_path_rate: float=0.1. The drop path rate. Default value is 0.1.
        layer_scale_init_value: float=0.0. The initial value of layer scale. Default value is 0.0.
        norm_layer: str='BN'. The type of normalization layer. Default value is 'BN'.
        act_layer: str='RELU'. The type of activation layer. Default value is 'RELU'.
        class_num: int=1000. The number of classes. Default value is 1000.
        fork_feat: bool=False. Whether to return feature maps. Default value is False.
        pretrained: str=None. The path of pretrained model. Default value is None.
        pconv_fw_type: str='split_cat'. The type of partial convolution forward. Default value is 'split_cat'.
        scale: float=1.0. The coefficient that controls the size of network parameters. 
    Returns:
        model: nn.Layer. Specific FasterNet model depends on args.
    """
    def __init__(self,
                 in_chans=3,
                 embed_dim=96,
                 depths=(1, 2, 8, 2),
                 mlp_ratio=2.0,
                 n_div=4,
                 patch_size=4,
                 patch_stride=4,
                 patch_size_t=2,
                 patch_stride_t=2,
                 patch_norm=True,
                 feature_dim=1280,
                 drop_path_rate=0.1,
                 layer_scale_init_value=0,
                 norm_layer='BN',
                 act_layer='RELU',
                 class_num=1000,
                 fork_feat=False,
                 pretrained=None,
                 pconv_fw_type='split_cat',
                 **kwargs):
        super().__init__()
        if norm_layer == 'BN':
            norm_layer = nn.BatchNorm2D
        else:
            raise NotImplementedError
        if act_layer == 'GELU':
            act_layer = nn.GELU
        elif act_layer == 'RELU':
            act_layer = nn.ReLU
        else:
            raise NotImplementedError
        if not fork_feat:
            self.class_num = class_num
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2**(self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        dpr = [
            x.item()
            for x in paddle.linspace(
                start=0, stop=drop_path_rate, num=sum(depths))
        ]
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(
                dim=int(embed_dim * 2**i_stage),
                n_div=n_div,
                depth=depths[i_stage],
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer,
                act_layer=act_layer,
                pconv_fw_type=pconv_fw_type)
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(
                    PatchMerging(
                        patch_size_t=patch_size_t,
                        patch_stride_t=patch_stride_t,
                        dim=int(embed_dim * 2**i_stage),
                        norm_layer=norm_layer))
        self.stages = nn.Sequential(*stages_list)
        self.avgpool_pre_head = nn.Sequential(
            nn.AdaptiveAvgPool2D(output_size=1),
            nn.Conv2D(
                in_channels=self.num_features,
                out_channels=feature_dim,
                kernel_size=1,
                bias_attr=False),
            act_layer())
        self.head = (nn.Linear(
            in_features=feature_dim, out_features=class_num)
                        if class_num > 0 else nn.Identity())
        self.apply(self.cls_init_weights)
        

    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.Conv1D, nn.Conv2D)):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.head(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def FasterNet_T0(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_T0"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_T0"], use_ssld)
    return model


def FasterNet_T1(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_T1"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_T1"], use_ssld)
    return model


def FasterNet_T2(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_T2"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_T2"], use_ssld)
    return model


def FasterNet_S(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_S"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_S"], use_ssld)
    return model


def FasterNet_M(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_M"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_M"], use_ssld)
    return model


def FasterNet_L(pretrained=False, use_ssld=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_L"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_L"], use_ssld)
    return model
