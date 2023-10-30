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

# Code was based on https://github.com/BR-IDL/PaddleViT/blob/develop/image_classification/MobileViT/mobilevit.py
# and https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
# reference: https://arxiv.org/abs/2110.02178

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import KaimingUniform, TruncatedNormal, Constant
import math

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileViT_XXS":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_XXS_pretrained.pdparams",
    "MobileViT_XS":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_XS_pretrained.pdparams",
    "MobileViT_S":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileViT_S_pretrained.pdparams",
}


def _init_weights_linear():
    weight_attr = ParamAttr(initializer=TruncatedNormal(std=.02))
    bias_attr = ParamAttr(initializer=Constant(0.0))
    return weight_attr, bias_attr


def _init_weights_layernorm():
    weight_attr = ParamAttr(initializer=Constant(1.0))
    bias_attr = ParamAttr(initializer=Constant(0.0))
    return weight_attr, bias_attr


class ConvBnAct(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1):
        super().__init__()
        self.in_channels = in_channels
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=ParamAttr(initializer=KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = nn.Silu()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class Identity(nn.Layer):
    """ Identity layer"""

    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class Mlp(nn.Layer):
    def __init__(self, embed_dim, mlp_ratio, dropout=0.1):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_linear()
        self.fc1 = nn.Linear(
            embed_dim,
            int(embed_dim * mlp_ratio),
            weight_attr=w_attr_1,
            bias_attr=b_attr_1)

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.fc2 = nn.Linear(
            int(embed_dim * mlp_ratio),
            embed_dim,
            weight_attr=w_attr_2,
            bias_attr=b_attr_2)

        self.act = nn.Silu()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 qkv_bias=True,
                 dropout=0.1,
                 attention_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.attn_head_dim = int(embed_dim / self.num_heads)
        self.all_head_dim = self.attn_head_dim * self.num_heads

        w_attr_1, b_attr_1 = _init_weights_linear()
        self.qkv = nn.Linear(
            embed_dim,
            self.all_head_dim * 3,
            weight_attr=w_attr_1,
            bias_attr=b_attr_1 if qkv_bias else False)

        self.scales = self.attn_head_dim**-0.5

        w_attr_2, b_attr_2 = _init_weights_linear()
        self.proj = nn.Linear(
            embed_dim, embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)

        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(axis=-1)

    def transpose_multihead(self, x):
        B, P, N, d = x.shape
        x = x.reshape([B, P, N, self.num_heads, d // self.num_heads])
        x = x.transpose([0, 1, 3, 2, 4])
        return x

    def forward(self, x):
        b_sz, n_patches, in_channels = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape([
            b_sz, n_patches, 3, self.num_heads,
            qkv.shape[-1] // self.num_heads // 3
        ])
        qkv = qkv.transpose([0, 3, 2, 1, 4])
        query, key, value = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        query = query * self.scales
        key = key.transpose([0, 1, 3, 2])
        # QK^T
        attn = paddle.matmul(query, key)
        attn = self.softmax(attn)
        attn = self.attn_dropout(attn)
        # weighted sum
        out = paddle.matmul(attn, value)
        out = out.transpose([0, 2, 1, 3]).reshape(
            [b_sz, n_patches, out.shape[1] * out.shape[3]])
        out = self.proj(out)
        out = self.proj_dropout(out)
        return out


class EncoderLayer(nn.Layer):
    def __init__(self,
                 embed_dim,
                 num_heads=4,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        w_attr_1, b_attr_1 = _init_weights_layernorm()
        w_attr_2, b_attr_2 = _init_weights_layernorm()

        self.attn_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1)
        self.attn = Attention(embed_dim, num_heads, qkv_bias, dropout,
                              attention_dropout)
        self.drop_path = DropPath(droppath) if droppath > 0. else Identity()
        self.mlp_norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_2, bias_attr=b_attr_2)
        self.mlp = Mlp(embed_dim, mlp_ratio, dropout)

    def forward(self, x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = self.drop_path(x)
        x = h + x
        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class Transformer(nn.Layer):
    """Transformer block for MobileViTBlock"""

    def __init__(self,
                 embed_dim,
                 num_heads,
                 depth,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 attention_dropout=0.,
                 droppath=0.):
        super().__init__()
        depth_decay = [x.item() for x in paddle.linspace(0, droppath, depth)]

        layer_list = []
        for i in range(depth):
            layer_list.append(
                EncoderLayer(embed_dim, num_heads, qkv_bias, mlp_ratio,
                             dropout, attention_dropout, droppath))
        self.layers = nn.LayerList(layer_list)

        w_attr_1, b_attr_1 = _init_weights_layernorm()
        self.norm = nn.LayerNorm(
            embed_dim, weight_attr=w_attr_1, bias_attr=b_attr_1, epsilon=1e-6)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        out = self.norm(x)
        return out


class MobileV2Block(nn.Layer):
    """Mobilenet v2 InvertedResidual block"""

    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expansion))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expansion != 1:
            layers.append(ConvBnAct(inp, hidden_dim, kernel_size=1))

        layers.extend([
            # dw
            ConvBnAct(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                padding=1),
            # pw-linear
            nn.Conv2D(
                hidden_dim, oup, 1, 1, 0, bias_attr=False),
            nn.BatchNorm2D(oup),
        ])

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileViTBlock(nn.Layer):
    """ MobileViTBlock for MobileViT"""

    def __init__(self,
                 dim,
                 hidden_dim,
                 depth,
                 num_heads=4,
                 qkv_bias=True,
                 mlp_ratio=2.0,
                 dropout=0.1,
                 attention_dropout=0.,
                 droppath=0.0,
                 patch_size=(2, 2)):
        super().__init__()
        self.patch_h, self.patch_w = patch_size

        # local representations
        self.conv1 = ConvBnAct(dim, dim, padding=1)
        self.conv2 = nn.Conv2D(
            dim, hidden_dim, kernel_size=1, stride=1, bias_attr=False)
        # global representations
        self.transformer = Transformer(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            depth=depth,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            attention_dropout=attention_dropout,
            droppath=droppath)

        # fusion
        self.conv3 = ConvBnAct(hidden_dim, dim, kernel_size=1)
        self.conv4 = ConvBnAct(2 * dim, dim, padding=1)

    def forward(self, x):
        h = x
        x = self.conv1(x)
        x = self.conv2(x)

        patch_h = self.patch_h
        patch_w = self.patch_w
        patch_area = int(patch_w * patch_h)
        _, in_channels, orig_h, orig_w = x.shape
        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False

        if new_w != orig_w or new_h != orig_h:
            x = F.interpolate(x, size=[new_h, new_w], mode="bilinear")
            interpolate = True

        num_patch_w, num_patch_h = new_w // patch_w, new_h // patch_h
        num_patches = num_patch_h * num_patch_w
        reshaped_x = x.reshape([-1, patch_h, num_patch_w, patch_w])
        transposed_x = reshaped_x.transpose([0, 2, 1, 3])
        reshaped_x = transposed_x.reshape(
            [-1, in_channels, num_patches, patch_area])
        transposed_x = reshaped_x.transpose([0, 3, 2, 1])

        x = transposed_x.reshape([-1, num_patches, in_channels])
        x = self.transformer(x)
        x = x.reshape([-1, patch_h * patch_w, num_patches, in_channels])

        _, pixels, num_patches, channels = x.shape
        x = x.transpose([0, 3, 2, 1])
        x = x.reshape([-1, num_patch_w, patch_h, patch_w])
        x = x.transpose([0, 2, 1, 3])
        x = x.reshape(
            [-1, channels, num_patch_h * patch_h, num_patch_w * patch_w])

        if interpolate:
            x = F.interpolate(x, size=[orig_h, orig_w])
        x = self.conv3(x)
        x = paddle.concat((h, x), axis=1)
        x = self.conv4(x)
        return x


class MobileViT(nn.Layer):
    """ MobileViT
        A PaddlePaddle impl of : `MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer`  -
          https://arxiv.org/abs/2110.02178
    """

    def __init__(self,
                 in_channels=3,
                 dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
                 hidden_dims=[96, 120, 144],
                 mv2_expansion=4,
                 class_num=1000):
        super().__init__()
        self.conv3x3 = ConvBnAct(
            in_channels, dims[0], kernel_size=3, stride=2, padding=1)
        self.mv2_block_1 = MobileV2Block(
            dims[0], dims[1], expansion=mv2_expansion)
        self.mv2_block_2 = MobileV2Block(
            dims[1], dims[2], stride=2, expansion=mv2_expansion)
        self.mv2_block_3 = MobileV2Block(
            dims[2], dims[3], expansion=mv2_expansion)
        self.mv2_block_4 = MobileV2Block(
            dims[3], dims[4], expansion=mv2_expansion)

        self.mv2_block_5 = MobileV2Block(
            dims[4], dims[5], stride=2, expansion=mv2_expansion)
        self.mvit_block_1 = MobileViTBlock(dims[5], hidden_dims[0], depth=2)

        self.mv2_block_6 = MobileV2Block(
            dims[5], dims[6], stride=2, expansion=mv2_expansion)
        self.mvit_block_2 = MobileViTBlock(dims[6], hidden_dims[1], depth=4)

        self.mv2_block_7 = MobileV2Block(
            dims[6], dims[7], stride=2, expansion=mv2_expansion)
        self.mvit_block_3 = MobileViTBlock(dims[7], hidden_dims[2], depth=3)
        self.conv1x1 = ConvBnAct(dims[7], dims[8], kernel_size=1)

        self.pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(dims[8], class_num)

    def forward(self, x):
        x = self.conv3x3(x)
        x = self.mv2_block_1(x)
        x = self.mv2_block_2(x)
        x = self.mv2_block_3(x)
        x = self.mv2_block_4(x)

        x = self.mv2_block_5(x)
        x = self.mvit_block_1(x)

        x = self.mv2_block_6(x)
        x = self.mvit_block_2(x)

        x = self.mv2_block_7(x)
        x = self.mvit_block_3(x)
        x = self.conv1x1(x)

        x = self.pool(x)
        x = x.reshape(x.shape[:2])

        x = self.dropout(x)
        x = self.linear(x)
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


def MobileViT_XXS(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 16, 24, 24, 24, 48, 64, 80, 320],
        hidden_dims=[64, 80, 96],
        mv2_expansion=2,
        **kwargs)

    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViT_XXS"], use_ssld=use_ssld)
    return model


def MobileViT_XS(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 32, 48, 48, 48, 64, 80, 96, 384],
        hidden_dims=[96, 120, 144],
        mv2_expansion=4,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViT_XS"], use_ssld=use_ssld)
    return model


def MobileViT_S(pretrained=False, use_ssld=False, **kwargs):
    model = MobileViT(
        in_channels=3,
        dims=[16, 32, 64, 64, 64, 96, 128, 160, 640],
        hidden_dims=[144, 192, 240],
        mv2_expansion=4,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["MobileViT_S"], use_ssld=use_ssld)
    return model
