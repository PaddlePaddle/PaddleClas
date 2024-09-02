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

import sys
import paddle
import os
from functools import partial
from typing import List
import copy
import math
import warnings
from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity

MODEL_URLS = {
    "FasterNet_T0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_T0_pretrained.pdparams",
    "FasterNet_T1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_T1_pretrained.pdparams",
    "FasterNet_T2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_T2_pretrained.pdparams",
    "FasterNet_S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_S_pretrained.pdparams",
    "FasterNet_M":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_M_pretrained.pdparams",
    "FasterNet_L":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/legendary_models/FasterNet_L_pretrained.pdparams",
}

__all__ = MODEL_URLS.keys()

NET_CONFIG = {
    "FasterNet_T0": [
        3, 40, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.0, 0, 'BN', 'GELU'
    ],
    "FasterNet_T1": [
        3, 64, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.02, 0, 'BN', 'GELU'
    ],
    "FasterNet_T2": [
        3, 96, [1, 2, 8, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.05, 0, 'BN', 'RELU'
    ],
    "FasterNet_S": [
        3, 128, [1, 2, 13, 2], 2, 4, 4, 4, 2, 2, True, 1280, 0.1, 0, 'BN', 'RELU'
    ],
    "FasterNet_M": [
        3, 144, [3, 4, 18, 3], 2, 4, 4, 4, 2, 2, True, 1280, 0.2, 0, 'BN', 'RELU'
    ],
    "FasterNet_L": [
        3, 192, [3, 4, 18, 3], 2, 4, 4, 4, 2, 2, True, 1280, 0.3, 0, 'BN', 'RELU'
    ],
}


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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
    if mean < a - 2 * std or mean > b + 2 * std:
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.'
            , stacklevel=2)
    with paddle.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(min=2 * l - 1, max=2 * u - 1)
        tensor.erfinv_()
        tensor.multiply_(y=paddle.to_tensor(std * math.sqrt(2.0)))
        tensor.add_(y=paddle.to_tensor(mean))
        tensor.clip_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\\text{mean}, \\text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \\text{mean} \\leq b`.

    NOTE: this impl is similar to the PyTorch trunc_normal_, the bounds [a, b] are
    applied while sampling the normal with mean/std applied, therefore a, b args
    should be adjusted to match the range of mean, std args.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Partial_conv3(paddle.nn.Layer):
    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = paddle.nn.Conv2D(in_channels=self.dim_conv3,
            out_channels=self.dim_conv3, kernel_size=3, stride=1, padding=1,
            bias_attr=False)
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: paddle.Tensor) ->paddle.Tensor:
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.
            dim_conv3, :, :])
        return x

    def forward_split_cat(self, x: paddle.Tensor) ->paddle.Tensor:
        x1, x2 = paddle.split(x=x, num_or_sections=[self.dim_conv3,
            self.dim_untouched], axis=1)
        x1 = self.partial_conv3(x1)
        x = paddle.concat(x=(x1, x2), axis=1)
        return x


class MLPBlock(paddle.nn.Layer):
    def __init__(self, dim, n_div, mlp_ratio, drop_path,
        layer_scale_init_value, act_layer, norm_layer, pconv_fw_type):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.drop_path = DropPath(drop_path) if drop_path > 0. else paddle.nn.Identity()
        self.n_div = n_div
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_layer: List[paddle.nn.Layer] = [paddle.nn.Conv2D(in_channels=
            dim, out_channels=mlp_hidden_dim, kernel_size=1, bias_attr=
            False), norm_layer(mlp_hidden_dim), act_layer(), paddle.nn.
            Conv2D(in_channels=mlp_hidden_dim, out_channels=dim,
            kernel_size=1, bias_attr=False)]
        self.mlp = paddle.nn.Sequential(*mlp_layer)
        self.spatial_mixing = Partial_conv3(dim, n_div, pconv_fw_type)
        if layer_scale_init_value > 0:
            self.layer_scale = (paddle.base.framework.EagerParamBase.
                from_tensor(tensor=layer_scale_init_value * paddle.ones(
                shape=dim), trainable=True))
            self.forward = self.forward_layer_scale
        else:
            self.forward = self.forward

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        shortcut = x        
        x = self.spatial_mixing(x)        
        x1 = self.mlp(x)
        x = shortcut + self.drop_path(self.mlp(x))
        return x

    def forward_layer_scale(self, x: paddle.Tensor) ->paddle.Tensor:
        shortcut = x
        x = self.spatial_mixing(x)
        x = shortcut + self.drop_path(self.layer_scale.unsqueeze(axis=-1).
            unsqueeze(axis=-1) * self.mlp(x))
        return x


class BasicStage(paddle.nn.Layer):
    def __init__(self, dim, depth, n_div, mlp_ratio, drop_path,
        layer_scale_init_value, norm_layer, act_layer, pconv_fw_type):
        super().__init__()
        blocks_list = [MLPBlock(dim=dim, n_div=n_div, mlp_ratio=mlp_ratio,
            drop_path=drop_path[i], layer_scale_init_value=
            layer_scale_init_value, norm_layer=norm_layer, act_layer=
            act_layer, pconv_fw_type=pconv_fw_type) for i in range(depth)]
        self.blocks = paddle.nn.Sequential(*blocks_list)

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.blocks(x)
        return x


class PatchEmbed(paddle.nn.Layer):
    def __init__(self, patch_size, patch_stride, in_chans, embed_dim,
        norm_layer):
        super().__init__()
        self.proj = paddle.nn.Conv2D(in_channels=in_chans, out_channels=
            embed_dim, kernel_size=patch_size, stride=patch_stride,
            bias_attr=False)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim,momentum=0.1)
        else:
            self.norm = paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.norm(self.proj(x))
        return x


class PatchMerging(paddle.nn.Layer):
    def __init__(self, patch_size2, patch_stride2, dim, norm_layer):
        super().__init__()
        self.reduction = paddle.nn.Conv2D(in_channels=dim, out_channels=2 *
            dim, kernel_size=patch_size2, stride=patch_stride2, bias_attr=False
            )
        if norm_layer is not None:
            self.norm = norm_layer(2 * dim)
        else:
            self.norm = paddle.nn.Identity()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.norm(self.reduction(x))
        return x


class FasterNet(paddle.nn.Layer):
    def __init__(self, in_chans=3, embed_dim=96, depths=(
        1, 2, 8, 2), mlp_ratio=2.0, n_div=4, patch_size=4, patch_stride=4,
        patch_size2=2, patch_stride2=2, patch_norm=True, feature_dim=1280,
        drop_path_rate=0.1, layer_scale_init_value=0, norm_layer='BN',
        act_layer='RELU', class_num=1000, fork_feat=False, init_cfg=None, pretrained=None,
        pconv_fw_type='split_cat', **kwargs):
        super().__init__()
        if norm_layer == 'BN':
            norm_layer = paddle.nn.BatchNorm2D
        else:
            raise NotImplementedError
        if act_layer == 'GELU':
            act_layer = paddle.nn.GELU
        elif act_layer == 'RELU':
            act_layer = paddle.nn.ReLU
        else:
            raise NotImplementedError
        if not fork_feat:
            self.class_num = class_num
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_stages - 1))
        self.mlp_ratio = mlp_ratio
        self.depths = depths
        self.patch_embed = PatchEmbed(patch_size=patch_size, patch_stride=
            patch_stride, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        dpr = [x.item() for x in paddle.linspace(start=0, stop=
            drop_path_rate, num=sum(depths))]
        stages_list = []
        for i_stage in range(self.num_stages):
            stage = BasicStage(dim=int(embed_dim * 2 ** i_stage), n_div=
                n_div, depth=depths[i_stage], mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1
                ])], layer_scale_init_value=layer_scale_init_value,
                norm_layer=norm_layer, act_layer=act_layer, pconv_fw_type=
                pconv_fw_type)
            stages_list.append(stage)
            if i_stage < self.num_stages - 1:
                stages_list.append(PatchMerging(patch_size2=patch_size2,
                    patch_stride2=patch_stride2, dim=int(embed_dim * 2 **
                    i_stage), norm_layer=norm_layer))
        self.stages = paddle.nn.Sequential(*stages_list)
        self.fork_feat = fork_feat
        if self.fork_feat:
            self.forward = self.forward_det
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get('FORK_LAST3', None):
                    raise NotImplementedError
                else:
                    layer = norm_layer(int(embed_dim * 2 ** i_emb))
                layer_name = f'norm{i_layer}'
                self.add_sublayer(name=layer_name, sublayer=layer)
        else:
            self.forward = self.forward_cls
            self.avgpool_pre_head = paddle.nn.Sequential(paddle.nn.
                AdaptiveAvgPool2D(output_size=1), paddle.nn.Conv2D(
                in_channels=self.num_features, out_channels=feature_dim,
                kernel_size=1, bias_attr=False), act_layer())
            self.head = paddle.nn.Linear(in_features=feature_dim,
                out_features=class_num
                ) if class_num > 0 else paddle.nn.Identity()
        self.apply(self.cls_init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not
            None):
            self.init_weights()

    def cls_init_weights(self, m):
        if isinstance(m, paddle.nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.Conv1D, paddle.nn.Conv2D)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, (paddle.nn.LayerNorm, paddle.nn.GroupNorm)):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward_cls(self, x):
        x = self.patch_embed(x)
        x = self.stages(x)
        x = self.avgpool_pre_head(x)
        x = paddle.flatten(x=x, start_axis=1)
        x = self.head(x)
        return x

    def forward_det(self, x: paddle.Tensor) ->paddle.Tensor:
        x = self.patch_embed(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f'norm{idx}')
                x_out = norm_layer(x)
                outs.append(x_out)
        return outs


def FasterNet_T0(pretrained: bool=False, **kwargs) :    
    model = FasterNet(*NET_CONFIG["FasterNet_T0"], **kwargs)
    _load_pretrained(pretrained, model, model_url=None, use_ssld=False)
    return model


def FasterNet_T1(pretrained: bool=False, use_ssld: bool=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_T1"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_T1"], use_ssld)
    return model


def FasterNet_T2(pretrained: bool=False, use_ssld: bool=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_T2"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_T2"], use_ssld)
    return model


def FasterNet_S(pretrained: bool=False, use_ssld: bool=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_S"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_S"], use_ssld)
    return model


def FasterNet_M(pretrained: bool=False, use_ssld: bool=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_M"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_M"], use_ssld)
    return model
    

def FasterNet_L(pretrained: bool=False, use_ssld: bool=False, **kwargs):
    model = FasterNet(*NET_CONFIG["FasterNet_L"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["FasterNet_L"], use_ssld)
    return model
