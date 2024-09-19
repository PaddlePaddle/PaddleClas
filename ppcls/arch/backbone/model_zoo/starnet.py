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

# reference: https://arxiv.org/abs/2403.19967

import paddle
import paddle.nn as nn

from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import DropPath

MODEL_URLS = {
    "StarNet_S1":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/StarNet_S1_pretrained.pdparams",
    "StarNet_S2":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/StarNet_S2_pretrained.pdparams",
    "StarNet_S3":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/StarNet_S3_pretrained.pdparams",
    "StarNet_S4":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/StarNet_S4_pretrained.pdparams",
}

__all__ = MODEL_URLS.keys()

NET_CONFIG = {
    "StarNet_S1": [24, [2, 2, 8, 3]],
    "StarNet_S2": [32, [1, 2, 6, 2]],
    "StarNet_S3": [32, [2, 2, 8, 4]],
    "StarNet_S4": [32, [3, 3, 12, 5]],
}


class ConvBN(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 with_bn=True):
        super().__init__()
        self.add_sublayer(
            name='conv',
            sublayer=nn.Conv2D(
                in_channels=in_planes,
                out_channels=out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups))
        if with_bn:
            self.add_sublayer(
                name='bn', sublayer=nn.BatchNorm2D(num_features=out_planes))
            init_Constant = nn.initializer.Constant(value=1)
            init_Constant(self.bn.weight)
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(self.bn.bias)


class Block(nn.Layer):
    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        self.dwconv = ConvBN(
            dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(
            dim, dim, 7, 1, (7 - 1) // 2, groups=dim, with_bn=False)
        self.act = nn.ReLU6()
        self.drop_path = (DropPath(drop_path)
                          if drop_path > 0. else nn.Identity())

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError("pretrained type is not available. ")


class StarNet(nn.Layer):
    """
    StarNet: StarNet for Image Classification
    Args:
        base_dim: int, base dimension of the model, default 32.
        depths: list, number of blocks in each stage, default [3, 3, 12, 5].
        mlp_ratio: int, ratio of hidden dim to mlp_dim, default 4.
        drop_path_rate: float, default 0.0, stochastic depth rate.
        class_num: int, default 1000, number of classes.
    """
    def __init__(self,
                 base_dim=32,
                 depths=[3, 3, 12, 5],
                 mlp_ratio=4,
                 drop_path_rate=0.0,
                 class_num=1000,
                 **kwargs):
        super().__init__()
        self.class_num = class_num
        self.in_channel = 32
        self.stem = nn.Sequential(
            ConvBN(
                3, self.in_channel, kernel_size=3, stride=2, padding=1),
            nn.ReLU6())
        dpr = [
            x.item()
            for x in paddle.linspace(
                start=0, stop=drop_path_rate, num=sum(depths))
        ]
        self.stages = nn.LayerList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2**i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [
                Block(self.in_channel, mlp_ratio, dpr[cur + i])
                for i in range(depths[i_layer])
            ]
            cur += depths[i_layer]
            self.stages.append(nn.Sequential(down_sampler, *blocks))
        self.norm = nn.BatchNorm2D(num_features=self.in_channel)
        self.avgpool = nn.AdaptiveAvgPool2D(output_size=1)
        self.head = nn.Linear(
            in_features=self.in_channel, out_features=class_num)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2D):
            pass
            if isinstance(m, nn.Linear) and m.bias is not None:
                init_Constant = nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, nn.LayerNorm or nn.BatchNorm2D):
            init_Constant = nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = paddle.flatten(x=self.avgpool(self.norm(x)), start_axis=1)
        return self.head(x)


def StarNet_S1(pretrained=False, use_ssld=False, **kwargs):
    model = StarNet(*NET_CONFIG["StarNet_S1"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["StarNet_S1"], use_ssld)
    return model


def StarNet_S2(pretrained=False, use_ssld=False, **kwargs):
    model = StarNet(*NET_CONFIG["StarNet_S2"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["StarNet_S2"], use_ssld)
    return model


def StarNet_S3(pretrained=False, use_ssld=False, **kwargs):
    model = StarNet(*NET_CONFIG["StarNet_S3"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["StarNet_S3"], use_ssld)
    return model


def StarNet_S4(pretrained=False, use_ssld=False, **kwargs):
    model = StarNet(*NET_CONFIG["StarNet_S4"], **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["StarNet_S4"], use_ssld)
    return model
