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
# Code was heavily based on https://github.com/microsoft/CvT
# reference: https://arxiv.org/abs/2103.15808

import paddle
import paddle.nn as nn
from paddle.nn.initializer import XavierUniform, TruncatedNormal, Constant

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "CvT_13_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_13_224_pretrained.pdparams",
    "CvT_13_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_13_384_pretrained.pdparams",
    "CvT_21_224":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_21_224_pretrained.pdparams",
    "CvT_21_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_21_384_pretrained.pdparams",
    "CvT_W24_384":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/CvT_W24_384_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())

xavier_uniform_ = XavierUniform()
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
    keep_prob = paddle.to_tensor(1 - drop_prob)
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

    def extra_repr(self):
        return f'drop_prob={self.drop_prob:.3f}'


def rearrange(x, pattern, **axes_lengths):
    if 'b (h w) c -> b c h w' == pattern:
        b, _, c = x.shape
        h, w = axes_lengths.pop('h', -1), axes_lengths.pop('w', -1)
        return x.transpose([0, 2, 1]).reshape([b, c, h, w])
    if 'b c h w -> b (h w) c' == pattern:
        b, c, h, w = x.shape
        return x.reshape([b, c, h * w]).transpose([0, 2, 1])
    if 'b t (h d) -> b h t d' == pattern:
        b, t, h_d = x.shape
        h = axes_lengths['h']
        return x.reshape([b, t, h, h_d // h]).transpose([0, 2, 1, 3])
    if 'b h t d -> b t (h d)' == pattern:
        b, h, t, d = x.shape
        return x.transpose([0, 2, 1, 3]).reshape([b, t, h * d])

    raise NotImplementedError(
        f"Rearrangement '{pattern}' has not been implemented.")


class Rearrange(nn.Layer):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths

    def forward(self, x):
        return rearrange(x, self.pattern, **self.axes_lengths)

    def extra_repr(self):
        return self.pattern


class QuickGELU(nn.Layer):
    def forward(self, x):
        return x * nn.functional.sigmoid(1.702 * x)


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
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 qkv_bias=False,
                 attn_drop=0.,
                 proj_drop=0.,
                 method='dw_bn',
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv=1,
                 padding_q=1,
                 with_cls_token=True,
                 **kwargs):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in, dim_out, kernel_size, padding_q, stride_q, 'linear'
            if method == 'avg' else method)
        self.conv_proj_k = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)
        self.conv_proj_v = self._build_projection(
            dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)

        self.proj_q = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias_attr=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride,
                          method):
        if method == 'dw_bn':
            proj = nn.Sequential(
                ('conv', nn.Conv2D(
                    dim_in,
                    dim_in,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias_attr=False,
                    groups=dim_in)), ('bn', nn.BatchNorm2D(dim_in)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')))
        elif method == 'avg':
            proj = nn.Sequential(
                ('avg', nn.AvgPool2D(
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    ceil_mode=True)),
                ('rearrage', Rearrange('b c h w -> b (h w) c')))
        elif method == 'linear':
            proj = None
        else:
            raise ValueError('Unknown method ({})'.format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = paddle.split(x, [1, h * w], 1)

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = rearrange(x, 'b c h w -> b (h w) c')

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = rearrange(x, 'b c h w -> b (h w) c')

        if self.with_cls_token:
            q = paddle.concat((cls_token, q), axis=1)
            k = paddle.concat((cls_token, k), axis=1)
            v = paddle.concat((cls_token, v), axis=1)

        return q, k, v

    def forward(self, x, h, w):
        if (self.conv_proj_q is not None or self.conv_proj_k is not None or
                self.conv_proj_v is not None):
            q, k, v = self.forward_conv(x, h, w)

        q = rearrange(self.proj_q(q), 'b t (h d) -> b h t d', h=self.num_heads)
        k = rearrange(self.proj_k(k), 'b t (h d) -> b h t d', h=self.num_heads)
        v = rearrange(self.proj_v(v), 'b t (h d) -> b h t d', h=self.num_heads)

        attn_score = (q @k.transpose([0, 1, 3, 2])) * self.scale
        attn = nn.functional.softmax(attn_score, axis=-1)
        attn = self.attn_drop(attn)

        x = attn @v
        x = rearrange(x, 'b h t d -> b t (h d)')

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Layer):
    def __init__(self,
                 dim_in,
                 dim_out,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        self.with_cls_token = kwargs['with_cls_token']

        self.norm1 = norm_layer(dim_in)
        self.attn = Attention(dim_in, dim_out, num_heads, qkv_bias, attn_drop,
                              drop, **kwargs)

        self.drop_path = DropPath(drop_path) \
            if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(in_features=dim_out,
                       hidden_features=dim_mlp_hidden,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvEmbed(nn.Layer):
    def __init__(self,
                 patch_size=7,
                 in_chans=3,
                 embed_dim=64,
                 stride=4,
                 padding=2,
                 norm_layer=None):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        if self.norm:
            x = self.norm(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x


class VisionTransformer(nn.Layer):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self,
                 patch_size=16,
                 patch_stride=16,
                 patch_padding=0,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            # img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer)

        with_cls_token = kwargs['with_cls_token']
        if with_cls_token:
            self.cls_token = self.create_parameter(
                shape=[1, 1, embed_dim], default_initializer=trunc_normal_)
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                Block(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs))
        self.blocks = nn.LayerList(blocks)

        if init == 'xavier':
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            xavier_uniform_(m.weight)
            if m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D)):
            zeros_(m.bias)
            ones_(m.weight)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand([B, -1, -1])
            x = paddle.concat((cls_tokens, x), axis=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = paddle.split(x, [1, H * W], 1)
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)

        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Layer):
    def __init__(self,
                 in_chans=3,
                 class_num=1000,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 init='trunc_norm',
                 spec=None):
        super().__init__()
        self.class_num = class_num

        self.num_stages = spec['NUM_STAGES']
        for i in range(self.num_stages):
            kwargs = {
                'patch_size': spec['PATCH_SIZE'][i],
                'patch_stride': spec['PATCH_STRIDE'][i],
                'patch_padding': spec['PATCH_PADDING'][i],
                'embed_dim': spec['DIM_EMBED'][i],
                'depth': spec['DEPTH'][i],
                'num_heads': spec['NUM_HEADS'][i],
                'mlp_ratio': spec['MLP_RATIO'][i],
                'qkv_bias': spec['QKV_BIAS'][i],
                'drop_rate': spec['DROP_RATE'][i],
                'attn_drop_rate': spec['ATTN_DROP_RATE'][i],
                'drop_path_rate': spec['DROP_PATH_RATE'][i],
                'with_cls_token': spec['CLS_TOKEN'][i],
                'method': spec['QKV_PROJ_METHOD'][i],
                'kernel_size': spec['KERNEL_QKV'][i],
                'padding_q': spec['PADDING_Q'][i],
                'padding_kv': spec['PADDING_KV'][i],
                'stride_kv': spec['STRIDE_KV'][i],
                'stride_q': spec['STRIDE_Q'][i],
            }

            stage = VisionTransformer(
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs)
            setattr(self, f'stage{i}', stage)

            in_chans = spec['DIM_EMBED'][i]

        dim_embed = spec['DIM_EMBED'][-1]
        self.norm = norm_layer(dim_embed)
        self.cls_token = spec['CLS_TOKEN'][-1]

        # Classifier head
        self.head = nn.Linear(dim_embed,
                              class_num) if class_num > 0 else nn.Identity()
        trunc_normal_(self.head.weight)

        bound = 1 / dim_embed**.5
        nn.initializer.Uniform(-bound, bound)(self.head.bias)

    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f'stage{i}.pos_embed')
            layers.add(f'stage{i}.cls_token')
        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f'stage{i}')(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = paddle.squeeze(x, axis=1)
        else:
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.norm(x)
            x = paddle.mean(x, axis=1)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


def _load_pretrained(pretrained,
                     model,
                     model_url,
                     use_ssld=False,
                     use_imagenet22kto1k_pretrained=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain(
            model,
            model_url,
            use_ssld=use_ssld,
            use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def CvT_13_224(pretrained=False, use_ssld=False, **kwargs):
    msvit_spec = dict(
        INIT='trunc_norm',
        NUM_STAGES=3,
        PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE=[4, 2, 2],
        PATCH_PADDING=[2, 1, 1],
        DIM_EMBED=[64, 192, 384],
        NUM_HEADS=[1, 3, 6],
        DEPTH=[1, 2, 10],
        MLP_RATIO=[4.0, 4.0, 4.0],
        ATTN_DROP_RATE=[0.0, 0.0, 0.0],
        DROP_RATE=[0.0, 0.0, 0.0],
        DROP_PATH_RATE=[0.0, 0.0, 0.1],
        QKV_BIAS=[True, True, True],
        CLS_TOKEN=[False, False, True],
        POS_EMBED=[False, False, False],
        QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV=[3, 3, 3],
        PADDING_KV=[1, 1, 1],
        STRIDE_KV=[2, 2, 2],
        PADDING_Q=[1, 1, 1],
        STRIDE_Q=[1, 1, 1])
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        init=msvit_spec.get('INIT', 'trunc_norm'),
        spec=msvit_spec,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["CvT_13_224"], use_ssld=use_ssld)
    return model


def CvT_13_384(pretrained=False,
               use_ssld=False,
               use_imagenet22kto1k_pretrained=False,
               **kwargs):
    msvit_spec = dict(
        INIT='trunc_norm',
        NUM_STAGES=3,
        PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE=[4, 2, 2],
        PATCH_PADDING=[2, 1, 1],
        DIM_EMBED=[64, 192, 384],
        NUM_HEADS=[1, 3, 6],
        DEPTH=[1, 2, 10],
        MLP_RATIO=[4.0, 4.0, 4.0],
        ATTN_DROP_RATE=[0.0, 0.0, 0.0],
        DROP_RATE=[0.0, 0.0, 0.0],
        DROP_PATH_RATE=[0.0, 0.0, 0.1],
        QKV_BIAS=[True, True, True],
        CLS_TOKEN=[False, False, True],
        POS_EMBED=[False, False, False],
        QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV=[3, 3, 3],
        PADDING_KV=[1, 1, 1],
        STRIDE_KV=[2, 2, 2],
        PADDING_Q=[1, 1, 1],
        STRIDE_Q=[1, 1, 1])
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        init=msvit_spec.get('INIT', 'trunc_norm'),
        spec=msvit_spec,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_13_384"],
        use_ssld=use_ssld,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def CvT_21_224(pretrained=False, use_ssld=False, **kwargs):
    msvit_spec = dict(
        INIT='trunc_norm',
        NUM_STAGES=3,
        PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE=[4, 2, 2],
        PATCH_PADDING=[2, 1, 1],
        DIM_EMBED=[64, 192, 384],
        NUM_HEADS=[1, 3, 6],
        DEPTH=[1, 4, 16],
        MLP_RATIO=[4.0, 4.0, 4.0],
        ATTN_DROP_RATE=[0.0, 0.0, 0.0],
        DROP_RATE=[0.0, 0.0, 0.0],
        DROP_PATH_RATE=[0.0, 0.0, 0.1],
        QKV_BIAS=[True, True, True],
        CLS_TOKEN=[False, False, True],
        POS_EMBED=[False, False, False],
        QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV=[3, 3, 3],
        PADDING_KV=[1, 1, 1],
        STRIDE_KV=[2, 2, 2],
        PADDING_Q=[1, 1, 1],
        STRIDE_Q=[1, 1, 1])
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        init=msvit_spec.get('INIT', 'trunc_norm'),
        spec=msvit_spec,
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["CvT_21_224"], use_ssld=use_ssld)
    return model


def CvT_21_384(pretrained=False,
               use_ssld=False,
               use_imagenet22kto1k_pretrained=False,
               **kwargs):
    msvit_spec = dict(
        INIT='trunc_norm',
        NUM_STAGES=3,
        PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE=[4, 2, 2],
        PATCH_PADDING=[2, 1, 1],
        DIM_EMBED=[64, 192, 384],
        NUM_HEADS=[1, 3, 6],
        DEPTH=[1, 4, 16],
        MLP_RATIO=[4.0, 4.0, 4.0],
        ATTN_DROP_RATE=[0.0, 0.0, 0.0],
        DROP_RATE=[0.0, 0.0, 0.0],
        DROP_PATH_RATE=[0.0, 0.0, 0.1],
        QKV_BIAS=[True, True, True],
        CLS_TOKEN=[False, False, True],
        POS_EMBED=[False, False, False],
        QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV=[3, 3, 3],
        PADDING_KV=[1, 1, 1],
        STRIDE_KV=[2, 2, 2],
        PADDING_Q=[1, 1, 1],
        STRIDE_Q=[1, 1, 1])
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        init=msvit_spec.get('INIT', 'trunc_norm'),
        spec=msvit_spec,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_21_384"],
        use_ssld=use_ssld,
        use_imagenet22kto1k_pretrained=use_imagenet22kto1k_pretrained)
    return model


def CvT_W24_384(pretrained=False, use_ssld=False, **kwargs):
    msvit_spec = dict(
        INIT='trunc_norm',
        NUM_STAGES=3,
        PATCH_SIZE=[7, 3, 3],
        PATCH_STRIDE=[4, 2, 2],
        PATCH_PADDING=[2, 1, 1],
        DIM_EMBED=[192, 768, 1024],
        NUM_HEADS=[3, 12, 16],
        DEPTH=[2, 2, 20],
        MLP_RATIO=[4.0, 4.0, 4.0],
        ATTN_DROP_RATE=[0.0, 0.0, 0.0],
        DROP_RATE=[0.0, 0.0, 0.0],
        DROP_PATH_RATE=[0.0, 0.0, 0.3],
        QKV_BIAS=[True, True, True],
        CLS_TOKEN=[False, False, True],
        POS_EMBED=[False, False, False],
        QKV_PROJ_METHOD=['dw_bn', 'dw_bn', 'dw_bn'],
        KERNEL_QKV=[3, 3, 3],
        PADDING_KV=[1, 1, 1],
        STRIDE_KV=[2, 2, 2],
        PADDING_Q=[1, 1, 1],
        STRIDE_Q=[1, 1, 1])
    model = ConvolutionalVisionTransformer(
        in_chans=3,
        act_layer=QuickGELU,
        init=msvit_spec.get('INIT', 'trunc_norm'),
        spec=msvit_spec,
        **kwargs)
    _load_pretrained(
        pretrained,
        model,
        MODEL_URLS["CvT_W24_384"],
        use_ssld=use_ssld,
        use_imagenet22kto1k_pretrained=True)
    return model
