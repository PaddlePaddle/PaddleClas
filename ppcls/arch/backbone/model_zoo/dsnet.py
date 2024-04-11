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

# reference: https://arxiv.org/abs/2105.14734v4

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from .vision_transformer import to_2tuple, zeros_, ones_, VisionTransformer, Identity, zeros_
from functools import partial
from paddle.nn.initializer import TruncatedNormal, Constant, Normal

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "DSNet_tiny":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DSNet_tiny_pretrained.pdparams",
    "DSNet_small":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DSNet_small_pretrained.pdparams",
    "DSNet_base":
    "https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/DSNet_base_pretrained.pdparams",
}

__all__ = list(MODEL_URLS.keys())


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
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Layer):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2D(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class DWConvMlp(nn.Layer):
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


def drop_path(x, drop_prob=0., training=False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
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


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        C = int(C // 3)
        qkv = x.reshape(
            (B, N, 3, self.num_heads, C // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2)))) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((B, N, C))
        x = self.proj_drop(x)
        return x


class Cross_Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, tokens_q, memory_k, memory_v, shape=None):
        assert shape is not None
        attn = (tokens_q.matmul(memory_k.transpose((0, 1, 3, 2)))) * self.scale
        attn = F.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(memory_v)).transpose((0, 2, 1, 3)).reshape(
            (shape[0], shape[1], shape[2]))
        x = self.proj_drop(x)
        return x


class MixBlock(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 downsample=2,
                 conv_ffn=False):
        super().__init__()
        self.pos_embed = nn.Conv2D(dim, dim, 3, padding=1, groups=dim)
        self.dim = dim
        self.norm1 = nn.BatchNorm2D(dim)
        self.conv1 = nn.Conv2D(dim, dim, 1)
        self.conv2 = nn.Conv2D(dim, dim, 1)
        self.dim_conv = int(dim * 0.5)
        self.dim_sa = dim - self.dim_conv
        self.norm_conv1 = nn.BatchNorm2D(self.dim_conv)
        self.norm_sa1 = nn.LayerNorm(self.dim_sa)
        self.conv = nn.Conv2D(
            self.dim_conv, self.dim_conv, 3, padding=1, groups=self.dim_conv)
        self.channel_up = nn.Linear(self.dim_sa, 3 * self.dim_sa)
        self.cross_channel_up_conv = nn.Conv2D(self.dim_conv,
                                               3 * self.dim_conv, 1)
        self.cross_channel_up_sa = nn.Linear(self.dim_sa, 3 * self.dim_sa)
        self.fuse_channel_conv = nn.Linear(self.dim_conv, self.dim_conv)
        self.fuse_channel_sa = nn.Linear(self.dim_sa, self.dim_sa)
        self.num_heads = num_heads
        self.attn = Attention(
            self.dim_sa,
            num_heads=self.num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.1,
            proj_drop=drop)
        self.cross_attn = Cross_Attention(
            self.dim_sa,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=0.1,
            proj_drop=drop)
        self.norm_conv2 = nn.BatchNorm2D(self.dim_conv)
        self.norm_sa2 = nn.LayerNorm(self.dim_sa)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = nn.BatchNorm2D(dim)
        self.downsample = downsample
        mlp_hidden_dim = int(dim * mlp_ratio)
        if conv_ffn:
            self.mlp = DWConvMlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop)
        else:
            self.mlp = Mlp(in_features=dim,
                           hidden_features=mlp_hidden_dim,
                           act_layer=act_layer,
                           drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        _, _, H, W = x.shape
        residual = x
        x = self.norm1(x)
        x = self.conv1(x)

        qkv = x[:, :self.dim_sa, :]
        conv = x[:, self.dim_sa:, :, :]
        residual_conv = conv
        conv = residual_conv + self.conv(self.norm_conv1(conv))

        sa = F.interpolate(
            qkv,
            size=(H // self.downsample, W // self.downsample),
            mode='bilinear')
        B, _, H_down, W_down = sa.shape
        sa = sa.flatten(2).transpose([0, 2, 1])
        residual_sa = sa
        sa = self.norm_sa1(sa)
        sa = self.channel_up(sa)
        sa = residual_sa + self.attn(sa)

        # cross attention
        residual_conv_co = conv
        residual_sa_co = sa
        conv_qkv = self.cross_channel_up_conv(self.norm_conv2(conv))
        conv_qkv = conv_qkv.flatten(2).transpose([0, 2, 1])

        sa_qkv = self.cross_channel_up_sa(self.norm_sa2(sa))

        B_conv, N_conv, C_conv = conv_qkv.shape
        C_conv = int(C_conv // 3)
        conv_qkv = conv_qkv.reshape((B_conv, N_conv, 3, self.num_heads,
                                     C_conv // self.num_heads)).transpose(
                                         (2, 0, 3, 1, 4))
        conv_q, conv_k, conv_v = conv_qkv[0], conv_qkv[1], conv_qkv[2]

        B_sa, N_sa, C_sa = sa_qkv.shape
        C_sa = int(C_sa // 3)
        sa_qkv = sa_qkv.reshape(
            (B_sa, N_sa, 3, self.num_heads, C_sa // self.num_heads)).transpose(
                (2, 0, 3, 1, 4))
        sa_q, sa_k, sa_v = sa_qkv[0], sa_qkv[1], sa_qkv[2]

        # sa -> conv
        conv = self.cross_attn(
            conv_q, sa_k, sa_v, shape=(B_conv, N_conv, C_conv))
        conv = self.fuse_channel_conv(conv)
        conv = conv.reshape((B, H, W, C_conv)).transpose((0, 3, 1, 2))
        conv = residual_conv_co + conv

        # conv -> sa
        sa = self.cross_attn(sa_q, conv_k, conv_v, shape=(B_sa, N_sa, C_sa))
        sa = residual_sa_co + self.fuse_channel_sa(sa)
        sa = sa.reshape((B, H_down, W_down, C_sa)).transpose((0, 3, 1, 2))
        sa = F.interpolate(sa, size=(H, W), mode='bilinear')
        x = paddle.concat([conv, sa], axis=1)
        x = residual + self.drop_path(self.conv2(x))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] //
                                                        patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class OverlapPatchEmbed(nn.Layer):
    """ Image to Overlapping Patch Embedding
    """

    def __init__(self,
                 img_size=224,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[
            1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2D(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        return x


class MixVisionTransformer(nn.Layer):
    """ Mixed Vision Transformer for DSNet
    A PaddlePaddle impl of : `Dual-stream Network for Visual Recognition` - https://arxiv.org/abs/2105.14734v4
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 class_num=1000,
                 embed_dim=[64, 128, 320, 512],
                 depth=[2, 2, 4, 1],
                 num_heads=[1, 2, 5, 8],
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=None,
                 overlap_embed=False,
                 conv_ffn=False):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            class_num (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Layer): normalization layer
            overlap_embed (bool): enable overlapped patch embedding if True
            conv_ffn (bool): enable depthwise convolution for mlp if True
        """
        super().__init__()
        self.class_num = class_num
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        downsamples = [8, 4, 2, 2]
        if overlap_embed:
            self.patch_embed1 = OverlapPatchEmbed(
                img_size=img_size,
                patch_size=7,
                stride=4,
                in_chans=in_chans,
                embed_dim=embed_dim[0])
            self.patch_embed2 = OverlapPatchEmbed(
                img_size=img_size // 4,
                patch_size=3,
                stride=2,
                in_chans=embed_dim[0],
                embed_dim=embed_dim[1])
            self.patch_embed3 = OverlapPatchEmbed(
                img_size=img_size // 8,
                patch_size=3,
                stride=2,
                in_chans=embed_dim[1],
                embed_dim=embed_dim[2])
            self.patch_embed4 = OverlapPatchEmbed(
                img_size=img_size // 16,
                patch_size=3,
                stride=2,
                in_chans=embed_dim[2],
                embed_dim=embed_dim[3])
        else:
            self.patch_embed1 = PatchEmbed(
                img_size=img_size,
                patch_size=4,
                in_chans=in_chans,
                embed_dim=embed_dim[0])
            self.patch_embed2 = PatchEmbed(
                img_size=img_size // 4,
                patch_size=2,
                in_chans=embed_dim[0],
                embed_dim=embed_dim[1])
            self.patch_embed3 = PatchEmbed(
                img_size=img_size // 8,
                patch_size=2,
                in_chans=embed_dim[1],
                embed_dim=embed_dim[2])
            self.patch_embed4 = PatchEmbed(
                img_size=img_size // 16,
                patch_size=2,
                in_chans=embed_dim[2],
                embed_dim=embed_dim[3])

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.mixture = False
        dpr = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depth))
        ]
        self.blocks1 = nn.LayerList([
            MixBlock(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                downsample=downsamples[0],
                conv_ffn=conv_ffn) for i in range(depth[0])
        ])

        self.blocks2 = nn.LayerList([
            MixBlock(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                downsample=downsamples[1],
                conv_ffn=conv_ffn) for i in range(depth[1])
        ])

        self.blocks3 = nn.LayerList([
            MixBlock(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                downsample=downsamples[2],
                conv_ffn=conv_ffn) for i in range(depth[2])
        ])

        if self.mixture:
            self.blocks4 = nn.LayerList([
                Block(
                    dim=embed_dim[3],
                    num_heads=16,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    downsample=downsamples[3],
                    conv_ffn=conv_ffn) for i in range(depth[3])
            ])
            self.norm = norm_layer(embed_dim[-1])
        else:
            self.blocks4 = nn.LayerList([
                MixBlock(
                    dim=embed_dim[3],
                    num_heads=num_heads[3],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    downsample=downsamples[3],
                    conv_ffn=conv_ffn) for i in range(depth[3])
            ])
            self.norm = nn.BatchNorm2D(embed_dim[-1])

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([('fc', nn.Linear(embed_dim, representation_size)),
                             ('act', nn.Tanh())]))
        else:
            self.pre_logits = Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1],
                              class_num) if class_num > 0 else Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            TruncatedNormal(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, class_num, global_pool=''):
        self.class_num = class_num
        self.head = nn.Linear(self.embed_dim,
                              class_num) if class_num > 0 else Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed1(x)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
        x = self.patch_embed4(x)
        if self.mixture:
            x = x.flatten(2).transpose([0, 2, 1])
        for blk in self.blocks4:
            x = blk(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.mixture:
            x = x.mean(1)
        else:
            x = x.flatten(2).mean(-1)
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


def DSNet_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = MixVisionTransformer(
        patch_size=16,
        depth=[2, 2, 4, 1],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6),
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["DSNet_tiny"], use_ssld=use_ssld)
    return model


def DSNet_small(pretrained=False, use_ssld=False, **kwargs):
    model = MixVisionTransformer(
        patch_size=16,
        depth=[3, 4, 8, 3],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6),
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["DSNet_small"], use_ssld=use_ssld)
    return model


def DSNet_base(pretrained=False, use_ssld=False, **kwargs):
    model = MixVisionTransformer(
        patch_size=16,
        depth=[3, 4, 28, 3],
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(
            nn.LayerNorm, eps=1e-6),
        **kwargs)
    _load_pretrained(
        pretrained, model, MODEL_URLS["DSNet_base"], use_ssld=use_ssld)
    return model
