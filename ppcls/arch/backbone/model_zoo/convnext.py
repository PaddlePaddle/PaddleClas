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
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/ConvNeXt_large_384_pretrained.pdparams",
    "ConvNeXtV2_atto":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_atto.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_base":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_base.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_femto":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_femto.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_huge":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_huge.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_large":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_large.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_nano":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_nano.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_pico":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_pico.fcmae_ft_in1k.pdparams",
    "ConvNeXtV2_tiny":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/convnextv2_tiny.fcmae_ft_in1k.pdparams"
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
        # Detect data format: (N, C, H, W) -> channels_first, (N, H, W, C) -> channels_last
        if x.shape[1] == self.normalized_shape[0]:
            # channels_first format (N, C, H, W)
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
        else:
            # channels_last format (N, H, W, C)
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight * x + self.bias
        return x


class LayerNorm2d(nn.Layer):
    def __init__(self, num_channels, epsilon=1e-6):
        super().__init__()
        self.weight = self.create_parameter(
            shape=[num_channels],
            default_initializer=ones_)
        self.bias = self.create_parameter(
            shape=[num_channels],
            default_initializer=zeros_)
        self.epsilon = epsilon
        self.num_channels = num_channels

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])  # NCHW -> NHWC
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)
        x = (x - mean) / paddle.sqrt(var + self.epsilon)
        x = x * self.weight + self.bias
        x = x.transpose([0, 3, 1, 2])  # NHWC -> NCHW
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


# ConvNeXt-V2


class GRN(nn.Layer):
    """Global Response Normalization (ConvNeXt-V2)."""

    def __init__(self, dim, eps=1e-6, channels_last=True):
        super().__init__()
        self.num_channels = dim
        self.eps = eps
        self.channels_last = channels_last
        self.gamma = self.create_parameter(
            shape=[dim],
            default_initializer=zeros_)
        self.beta = self.create_parameter(
            shape=[dim],
            default_initializer=zeros_)

    def forward(self, x):
        if self.channels_last:
            gx = paddle.sqrt((x * x).sum(axis=[1, 2], keepdim=True))
            nx = gx / (gx.mean(axis=-1, keepdim=True) + self.eps)
            w = self.gamma.reshape([1, 1, 1, -1])
            b = self.beta.reshape([1, 1, 1, -1])
            return x + b + w * (x * nx)
        else:
            gx = paddle.sqrt((x * x).sum(axis=[2, 3], keepdim=True))
            nx = gx / (gx.mean(axis=1, keepdim=True) + self.eps)
            w = self.gamma.reshape([1, -1, 1, 1])
            b = self.beta.reshape([1, -1, 1, 1])
            return x + b + w * (x * nx)


class ConvNeXtV2Block(nn.Layer):
    """ConvNeXt-V2 Block with GRN."""

    def __init__(self, dim, drop_path=0., use_grn=True, conv_mlp=False, channels_last=True):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, kernel_size=7, padding=3, groups=dim)
        self.conv_mlp = conv_mlp
        self.channels_last = channels_last

        if conv_mlp:
            self.norm = LayerNorm2d(dim, epsilon=1e-6)
        else:
            self.norm = ChannelsFirstLayerNorm(dim, epsilon=1e-6)

        if conv_mlp:
            self.pwconv1 = nn.Conv2D(dim, 4 * dim, kernel_size=1, bias_attr=True)
            self.pwconv2 = nn.Conv2D(4 * dim, dim, kernel_size=1, bias_attr=True)
        else:
            self.pwconv1 = nn.Linear(dim, 4 * dim)
            self.pwconv2 = nn.Linear(4 * dim, dim)

        self.act = nn.GELU()

        if use_grn:
            self.grn = GRN(4 * dim, eps=1e-6, channels_last=not conv_mlp)
        else:
            self.grn = None

        self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)

        if self.conv_mlp:
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            if self.grn is not None:
                x = self.grn(x)
            x = self.pwconv2(x)
        else:
            x = x.transpose([0, 2, 3, 1])
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            if self.grn is not None:
                x = self.grn(x)
            x = self.pwconv2(x)
            x = x.transpose([0, 3, 1, 2])

        x = input + self.drop_path(x)
        return x


class NormMlpClassifierHead(nn.Layer):
    """Classifier head: global_pool -> norm -> flatten -> drop -> fc."""
    
    def __init__(self, num_features, num_classes, drop_rate=0.):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2D(1)
        self.norm = LayerNorm2d(num_features, epsilon=1e-6)
        self.flatten = nn.Flatten()
        self.pre_logits = nn.Identity()
        self.drop = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.pre_logits(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class ConvNeXtV2(nn.Layer):
    """ConvNeXt-V2. Reference: https://arxiv.org/abs/2301.00808"""

    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.,
                 head_init_scale=1.,
                 use_grn=True,
                 conv_mlp=False):
        super().__init__()

        self.downsample_layers = nn.LayerList()
        stem = nn.Sequential(
            nn.Conv2D(in_chans, dims[0], 4, stride=4),
            ChannelsFirstLayerNorm(dims[0], epsilon=1e-6))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                ChannelsFirstLayerNorm(dims[i], epsilon=1e-6),
                nn.Conv2D(dims[i], dims[i + 1], 2, stride=2))
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.LayerList()
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(* [
                ConvNeXtV2Block(
                    dim=dims[i],
                    drop_path=dp_rates[cur + j],
                    use_grn=use_grn,
                    conv_mlp=conv_mlp,
                    channels_last=not conv_mlp)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm_pre = nn.Identity()
        self.head = NormMlpClassifierHead(
            num_features=dims[-1],
            num_classes=class_num,
            drop_rate=0.
        )

        self.apply(self._init_weights)
        self.head.fc.weight.set_value(self.head.fc.weight * head_init_scale)
        self.head.fc.bias.set_value(self.head.fc.bias * head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.norm_pre(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# ConvNeXt-V2 model variants


def ConvNeXtV2_atto(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[40, 80, 160, 320],
        use_grn=True,
        conv_mlp=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_atto"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_femto(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[48, 96, 192, 384],
        use_grn=True,
        conv_mlp=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_femto"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_pico(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        use_grn=True,
        conv_mlp=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_pico"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_nano(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[2, 2, 8, 2],
        dims=[80, 160, 320, 640],
        use_grn=True,
        conv_mlp=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_nano"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        use_grn=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_tiny"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_base(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        use_grn=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_base"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_large(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        use_grn=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_large"], use_ssld=use_ssld)
    return model


def ConvNeXtV2_huge(pretrained=False, use_ssld=False, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        use_grn=True,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["ConvNeXtV2_huge"], use_ssld=use_ssld)
    return model


__all__ = list(MODEL_URLS.keys())
