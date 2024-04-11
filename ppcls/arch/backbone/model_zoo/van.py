# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

# Code was heavily based on https://github.com/Visual-Attention-Network/VAN-Classification
# reference: https://arxiv.org/abs/2202.09741

from functools import partial
import math
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "VAN_B0":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B0_pretrained.pdparams",
    "VAN_B1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B1_pretrained.pdparams",
    "VAN_B2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B2_pretrained.pdparams",
    "VAN_B3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/VAN_B3_pretrained.pdparams"
}

__all__ = list(MODEL_URLS.keys())

trunc_normal_ = TruncatedNormal(std=.02)
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ...
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = paddle.to_tensor(1 - drop_prob, dtype=x.dtype)
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + paddle.rand(shape, dtype=x.dtype)
    random_tensor = paddle.floor(random_tensor)  # binarize
    output = x.divide(keep_prob) * random_tensor
    return output


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


@paddle.jit.not_to_static
def swapdim(x, dim1, dim2):
    a = list(range(len(x.shape)))
    a[dim1], a[dim2] = a[dim2], a[dim1]
    return x.transpose(a)


class Mlp(nn.Layer):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Layer):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2D(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2D(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2D(dim, dim, 1)

    def forward(self, x):
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)
        return x * attn


class Attention(nn.Layer):
    def __init__(self, d_model):
        super().__init__()
        self.proj_1 = nn.Conv2D(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2D(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 mlp_ratio=4.,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.BatchNorm2D(dim)
        self.attn = Attention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2D(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))
        self.layer_scale_2 = self.create_parameter(
            shape=[dim, 1, 1],
            default_initializer=Constant(value=layer_scale_init_value))

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2)
        self.norm = nn.BatchNorm2D(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Layer):
    r""" VAN
    A PaddlePaddle impl of : `Visual Attention Network`  -
      https://arxiv.org/pdf/2202.09741.pdf
    """

    def __init__(self,
                 img_size=224,
                 in_chans=3,
                 class_num=1000,
                 embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4],
                 drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3],
                 num_stages=4,
                 flag=False):
        super().__init__()
        if flag == False:
            self.class_num = class_num
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x for x in paddle.linspace(0, drop_path_rate, sum(depths))
               ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else img_size // (2**(i + 1)),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            block = nn.LayerList([
                Block(
                    dim=embed_dims[i],
                    mlp_ratio=mlp_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j]) for j in range(depths[i])
            ])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3],
                              class_num) if class_num > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, nn.Conv2D):
            fan_out = m._kernel_size[0] * m._kernel_size[1] * m._out_channels
            fan_out //= m._groups
            m.weight.set_value(
                paddle.normal(
                    std=math.sqrt(2.0 / fan_out), shape=m.weight.shape))
            if m.bias is not None:
                zeros_(m.bias)

    def forward_features(self, x):
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)

            x = x.flatten(2)
            x = swapdim(x, 1, 2)
            x = norm(x)
            if i != self.num_stages - 1:
                x = x.reshape([B, H, W, x.shape[2]]).transpose([0, 3, 1, 2])

        return x.mean(axis=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias_attr=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
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


def VAN_B0(pretrained=False, use_ssld=False, **kwargs):
    model = VAN(embed_dims=[32, 64, 160, 256],
                mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(
                    nn.LayerNorm, epsilon=1e-6),
                depths=[3, 3, 5, 2],
                **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["VAN_B0"], use_ssld=use_ssld)
    return model


def VAN_B1(pretrained=False, use_ssld=False, **kwargs):
    model = VAN(embed_dims=[64, 128, 320, 512],
                mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(
                    nn.LayerNorm, epsilon=1e-6),
                depths=[2, 2, 4, 2],
                **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["VAN_B1"], use_ssld=use_ssld)
    return model


def VAN_B2(pretrained=False, use_ssld=False, **kwargs):
    model = VAN(embed_dims=[64, 128, 320, 512],
                mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(
                    nn.LayerNorm, epsilon=1e-6),
                depths=[3, 3, 12, 3],
                **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["VAN_B2"], use_ssld=use_ssld)
    return model


def VAN_B3(pretrained=False, use_ssld=False, **kwargs):
    model = VAN(embed_dims=[64, 128, 320, 512],
                mlp_ratios=[8, 8, 4, 4],
                norm_layer=partial(
                    nn.LayerNorm, epsilon=1e-6),
                depths=[3, 5, 27, 3],
                **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["VAN_B3"], use_ssld=use_ssld)
    return model
