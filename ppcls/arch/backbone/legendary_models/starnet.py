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
from ....utils.save_load import load_dygraph_pretrain
from ..model_zoo.vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity

NET_CONFIG = {
    "Starnet_s1": [
        24, [2, 2, 8, 3]
    ],
    "Starnet_s2": [
        32, [1, 2, 6, 2]
    ],
    "Starnet_s3": [
        32, [2, 2, 8, 4]
    ],
    "Starnet_s4": [
        32, [3, 3, 12, 5]
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
        

class ConvBN(paddle.nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=1, stride=1,
        padding=0, dilation=1, groups=1, with_bn=True):
        super().__init__()
        self.add_sublayer(name='conv', sublayer=paddle.nn.Conv2D(
            in_channels=in_planes, out_channels=out_planes, kernel_size=
            kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups))
        if with_bn:
            self.add_sublayer(name='bn', sublayer=paddle.nn.BatchNorm2D(
                num_features=out_planes))
            init_Constant = paddle.nn.initializer.Constant(value=1)
            init_Constant(self.bn.weight)
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(self.bn.bias)


class Block(paddle.nn.Layer):

    def __init__(self, dim, mlp_ratio=3, drop_path=0.0):
        super().__init__()
        self.dwconv = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim,
            with_bn=True)
        self.f1 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.f2 = ConvBN(dim, mlp_ratio * dim, 1, with_bn=False)
        self.g = ConvBN(mlp_ratio * dim, dim, 1, with_bn=True)
        self.dwconv2 = ConvBN(dim, dim, 7, 1, (7 - 1) // 2, groups=dim,
            with_bn=False)
        self.act = paddle.nn.ReLU6()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else paddle.nn.Identity()
    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x1, x2 = self.f1(x), self.f2(x)
        x = self.act(x1) * x2
        x = self.dwconv2(self.g(x))
        x = input + self.drop_path(x)
        return x


class StarNet(paddle.nn.Layer):
    def __init__(self, base_dim=32, depths=[3, 3, 12, 5], mlp_ratio=4,
        drop_path_rate=0.0, num_classes=1000, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channel = 32
        self.stem = paddle.nn.Sequential(ConvBN(3, self.in_channel,
            kernel_size=3, stride=2, padding=1), paddle.nn.ReLU6())
        dpr = [x.item() for x in paddle.linspace(start=0, stop=
            drop_path_rate, num=sum(depths))]
        self.stages = paddle.nn.LayerList()
        cur = 0
        for i_layer in range(len(depths)):
            embed_dim = base_dim * 2 ** i_layer
            down_sampler = ConvBN(self.in_channel, embed_dim, 3, 2, 1)
            self.in_channel = embed_dim
            blocks = [Block(self.in_channel, mlp_ratio, dpr[cur + i]) for i in
                range(depths[i_layer])]
            cur += depths[i_layer]
            self.stages.append(paddle.nn.Sequential(down_sampler, *blocks))
        self.norm = paddle.nn.BatchNorm2D(num_features=self.in_channel)
        self.avgpool = paddle.nn.AdaptiveAvgPool2D(output_size=1)
        self.head = paddle.nn.Linear(in_features=self.in_channel,
            out_features=num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, paddle.nn.Linear or paddle.nn.Conv2D):
            pass
            if isinstance(m, paddle.nn.Linear) and m.bias is not None:
                init_Constant = paddle.nn.initializer.Constant(value=0)
                init_Constant(m.bias)
        elif isinstance(m, paddle.nn.LayerNorm or paddle.nn.BatchNorm2D):
            init_Constant = paddle.nn.initializer.Constant(value=0)
            init_Constant(m.bias)
            init_Constant = paddle.nn.initializer.Constant(value=1.0)
            init_Constant(m.weight)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = paddle.flatten(x=self.avgpool(self.norm(x)), start_axis=1)
        return self.head(x)


def Starnet_s1(pretrained: bool=False, **kwargs):
    model = StarNet(*NET_CONFIG["Starnet_s1"], **kwargs)
    _load_pretrained(pretrained, model, model_url=None, use_ssld=False)
    return model


def Starnet_s2(pretrained: bool=False, **kwargs):
    model = StarNet(*NET_CONFIG["Starnet_s2"], **kwargs)
    _load_pretrained(pretrained, model, model_url=None, use_ssld=False)
    return model


def Starnet_s3(pretrained: bool=False, **kwargs):
    model = StarNet(*NET_CONFIG["Starnet_s3"], **kwargs)
    _load_pretrained(pretrained, model, model_url=None, use_ssld=False)
    return model


def Starnet_s4(pretrained: bool=False, **kwargs):
    model = StarNet(*NET_CONFIG["Starnet_s4"], **kwargs)
    _load_pretrained(pretrained, model, model_url=None, use_ssld=False)
    return model
