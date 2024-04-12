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
#
# Code was heavily based on https://github.com/facebookresearch/ConvNeXt

import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "ConvNeXt_tiny":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_tiny_pretrained.pdparams",
    "ConvNeXt_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_small_pretrained.pdparams",
    "ConvNeXt_base_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_base_224_pretrained.pdparams",
    "ConvNeXt_base_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_base_384_pretrained.pdparams",
    "ConvNeXt_large_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_large_224_pretrained.pdparams",
    "ConvNeXt_large_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_large_384_pretrained.pdparams"
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


class ChannelsFirstLayerNorm(nn.Layer):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, epsilon=1e-5):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[normalized_shape], default_initializer=ones_)
        self.bias = self.create_parameter(
            shape=[normalized_shape], default_initializer=zeros_)
        self.epsilon = epsilon
        self.normalized_shape = [normalized_shape]

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / paddle.sqrt(s + self.epsilon)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class Block(nn.Layer):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, 7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, epsilon=1e-6)
        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        if layer_scale_init_value > 0:
            self.gamma = self.create_parameter(
                shape=[dim],
                default_initializer=Constant(value=layer_scale_init_value))
        else:
            self.gamma = None
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXt(nn.Layer):
    r""" ConvNeXt
        A PaddlePaddle impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        class_num (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.):
        super().__init__()

        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.LayerList()
        stem = nn.Sequential(
            nn.Conv2D(
                in_chans, dims[0], 4, stride=4),
            ChannelsFirstLayerNorm(
                dims[0], epsilon=1e-6))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                ChannelsFirstLayerNorm(
                    dims[i], epsilon=1e-6),
                nn.Conv2D(
                    dims[i], dims[i + 1], 2, stride=2), )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.LayerList()
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(* [
                Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer
        self.head = nn.Linear(dims[-1], class_num)

        self.apply(self._init_weights)
        self.head.weight.set_value(self.head.weight * head_init_scale)
        self.head.bias.set_value(self.head.bias * head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # global average pooling, (N, C, H, W) -> (N, C)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
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


def ConvNeXt_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_tiny"], use_ssld=use_ssld)
    return model


def ConvNeXt_small(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_small"], use_ssld=use_ssld)
    return model


def ConvNeXt_base_224(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_base_224"], use_ssld=use_ssld)
    return model


def ConvNeXt_base_384(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_base_384"], use_ssld=use_ssld)
    return model


def ConvNeXt_large_224(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_large_224"], use_ssld=use_ssld)
    return model


def ConvNeXt_large_384(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXt_large_384"], use_ssld=use_ssld)
    return model
