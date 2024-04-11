from paddle import ParamAttr
from paddle.nn.initializer import KaimingNormal
import numpy as np
import paddle
import paddle.nn as nn
from paddle.nn.initializer import TruncatedNormal, Constant, Normal
from paddle.nn import functional as F

trunc_normal_ = TruncatedNormal(std=.02)
normal_ = Normal
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)

def resize_pos_embed(pos_embed,
                     src_shape,
                     dst_shape,
                     mode='bicubic',
                     num_extra_tokens=1):
    """Resize pos_embed weights.

    Args:
        pos_embed (paddle.Tensor): Position embedding weights with shape
            [1, L, C].
        src_shape (tuple): The resolution of downsampled origin training
            image, in format (H, W).
        dst_shape (tuple): The resolution of downsampled new training
            image, in format (H, W).
        mode (str): Algorithm used for upsampling. Choose one from 'nearest',
            'linear', 'bilinear', 'bicubic' and 'trilinear'.
            Defaults to 'bicubic'.
        num_extra_tokens (int): The number of extra tokens, such as cls_token.
            Defaults to 1.

    Returns:
        paddle.Tensor: The resized pos_embed of shape [1, L_new, C]
    """
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, 'shape of pos_embed must be [1, L, C]'
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, \
        f"The length of `pos_embed` ({L}) doesn't match the expected " \
        f'shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the' \
        '`img_size` argument.'
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape([-1, src_h, src_w, C]).transpose(
        [0, 3, 1, 2])

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        paddle.cast(src_weight, paddle.float32),
        size=dst_shape,
        align_corners=False,
        mode=mode)
    dst_weight = paddle.flatten(dst_weight, 2).transpose([0, 2, 1])
    dst_weight = paddle.cast(dst_weight, src_weight.dtype)

    return paddle.concat((extra_tokens, dst_weight), axis=1)

def pading_for_not_divisible(pixel_values,
                             height,
                             width,
                             patch_size,
                             format="NCHW",
                             function="split"):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)
    if height % patch_size[0] == 0 and width % patch_size[1] == 0:
        return pixel_values, (0, 0, 0, 0, 0, 0, 0, 0)
    if function == "split":
        pading_width = patch_size[1] - width % patch_size[1]
        pading_height = patch_size[0] - height % patch_size[0]
    elif function == "merge":
        pading_width = width % 2
        pading_height = height % 2
    if format == "NCHW":
        pad_index = [0, 0, 0, 0, 0, pading_height, 0, pading_width]
    elif format == "NHWC":
        pad_index = [0, 0, 0, pading_height, 0, pading_width, 0, 0]
    else:
        assert ("vaild format")

    return F.pad(pixel_values, pad_index), pad_index

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


class ConvBNLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 bias_attr=False,
                 groups=1,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            weight_attr=paddle.ParamAttr(
                initializer=nn.initializer.KaimingUniform()),
            bias_attr=bias_attr)
        self.norm = nn.BatchNorm2D(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


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


class ConvMixer(nn.Layer):
    def __init__(
            self,
            dim,
            num_heads=8,
            HW=[8, 25],
            local_k=[3, 3], ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2D(
            dim,
            dim,
            local_k,
            1, [local_k[0] // 2, local_k[1] // 2],
            groups=num_heads,
            weight_attr=ParamAttr(initializer=KaimingNormal()))

    def forward(self, x, input_dimension):
        h, w = input_dimension
        x = x.transpose([0, 2, 1]).reshape([0, self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose([0, 2, 1])
        return x


class Attention(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads=8,
                 mixer='Global',
                 HW=None,
                 local_k=[7, 11],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias_attr=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        self.local_k = local_k
        self.mixer = mixer
    def get_mask(self,input_dimension):
        if self.HW is not None:
            H = input_dimension[0]
            W = input_dimension[1]
            self.N = H * W
            self.C = self.dim
        if self.mixer == 'Local' and self.HW is not None:
            hk = self.local_k[0]
            wk = self.local_k[1]
            mask = paddle.ones(
                [H * W, H + hk - 1, W + wk - 1], dtype='float32')
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.
            mask_paddle = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk //
                               2].flatten(1)
            mask_inf = paddle.full([H * W, H * W], '-inf', dtype='float32')
            mask = paddle.where(mask_paddle < 1, mask_paddle, mask_inf)
            return mask
        return None
    def forward(self, x, input_dimension):
        qkv = self.qkv(x).reshape(
            (0, -1, 3, self.num_heads, self.head_dim)).transpose(
                (2, 0, 3, 1, 4))
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]

        attn = (q.matmul(k.transpose((0, 1, 3, 2))))
        if self.mixer == 'Local':
            attn += self.get_mask(input_dimension)
        attn = nn.functional.softmax(attn, axis=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose((0, 2, 1, 3)).reshape((0, -1, self.dim))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Layer):
    def __init__(self,
                 dim,
                 num_heads,
                 mixer='Global',
                 local_mixer=[7, 11],
                 HW=None,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer='nn.LayerNorm',
                 epsilon=1e-6,
                 prenorm=True):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop)
        elif mixer == 'Conv':
            self.mixer = ConvMixer(
                dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)
        self.prenorm = prenorm

    def forward(self, x, input_dimension):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x,input_dimension)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x),input_dimension))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Layer):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 img_size=[32, 100],
                 in_channels=3,
                 embed_dim=768,
                 sub_num=2,
                 patch_size=[4, 4],
                 mode='pope'):
        super().__init__()
        num_patches = (img_size[1] // (2 ** sub_num)) * \
                      (img_size[0] // (2 ** sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.window_size = ((img_size[0] // (2 ** sub_num), (img_size[1] // (2 ** sub_num))))
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias_attr=None))
        elif mode == 'linear':
            self.proj = nn.Conv2D(
                1, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape

        x, _ = pading_for_not_divisible(x, H, W, self.patch_size, "BCHW")
        x = self.proj(x)
        _, _, height, width = x.shape
        output_dimensions = (height, width)
        x = x.flatten(2).transpose((0, 2, 1))
        return x, output_dimensions


class SubSample(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 types='Pool',
                 stride=[2, 1],
                 sub_norm='nn.LayerNorm',
                 act=None):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.maxpool = nn.MaxPool2D(
                kernel_size=[3, 5], stride=stride, padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                weight_attr=ParamAttr(initializer=KaimingNormal()))
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):

        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            output_dimension = (x.shape[2],x.shape[3])
            out = self.proj(x.flatten(2).transpose((0, 2, 1)))
        else:
            x = self.conv(x)
            output_dimension = (x.shape[2],x.shape[3])
            out = x.flatten(2).transpose((0, 2, 1))
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out, output_dimension


class SVTRNet(nn.Layer):
    def __init__(
            self,
            class_num=1000,
            img_size=[48, 320],
            in_channels=3,
            embed_dim=[192, 256, 512],
            depth=[6, 6, 9],
            num_heads=[6, 8, 16],
            mixer=['Conv'] * 9 + ['Global'] *
            12,  # Local atten, Global atten, Conv
            local_mixer=[[5, 5], [5, 5], [5, 5]],
            patch_merging='Conv',  # Conv, Pool, None
            mlp_ratio=4,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            last_drop=0.1,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer='nn.LayerNorm',
            sub_norm='nn.LayerNorm',
            epsilon=1e-6,
            out_channels=512,
            out_char_num=40,
            block_unit='Block',
            act='nn.GELU',
            last_stage=False,
            sub_num=2,
            prenorm=True,
            use_lenhead=False,
            **kwargs):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num)
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.pos_embed = self.create_parameter(
            shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        self.add_parameter("pos_embed", self.pos_embed)
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.LayerList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.LayerList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=[2, 1],
                types=patch_merging)
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.LayerList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                epsilon=epsilon,
                prenorm=prenorm) for i in range(depth[2])
        ])
        self.flatten = nn.Flatten(start_axis=0, stop_axis=1)
        self.fc = nn.Linear(embed_dim[2], class_num)

        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2D([1, out_char_num])
            self.last_conv = nn.Conv2D(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop, mode="downscale_in_infer")
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(
                p=last_drop, mode="downscale_in_infer")

        trunc_normal_(self.pos_embed)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    def forward_features(self, x):
        x,output_dimensions = self.patch_embed(x)
        x = x + resize_pos_embed(self.pos_embed,self.patch_embed.window_size,output_dimensions,num_extra_tokens=0)
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x, output_dimensions)
        if self.patch_merging is not None:
            x, output_dimensions = self.sub_sample1(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[0], output_dimensions[0], output_dimensions[1]]))
        for blk in self.blocks2:
            x = blk(x, output_dimensions)
        if self.patch_merging is not None:
            x, output_dimensions = self.sub_sample2(
                x.transpose([0, 2, 1]).reshape(
                    [0, self.embed_dim[1], output_dimensions[0], output_dimensions[1]]))
        for blk in self.blocks3:
            x = blk(x, output_dimensions)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.mean(1)
        x = self.fc(x)
        return x


def SVTR_tiny(pretrained=False, use_ssld=False, **kwargs):
    model = SVTRNet(
        img_size=[48, 320],
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer=['Conv'] * 6 + ['Global'] * 6,
        local_mixer=[[5, 5], [5, 5], [5, 5]],
        mlp_ratio=4,
        qkv_bias=True,
        out_channels=256,
        out_char_num=40,
        epsilon=1e-6,
        **kwargs)
    return model


def SVTR_base(pretrained=False, use_ssld=False, **kwargs):
    model = SVTRNet(
        img_size=[48, 320],
        embed_dim=[128, 256, 384],
        depth=[6, 6, 6],
        num_heads=[4, 8, 12],
        mixer=['Conv'] * 9 + ['Global'] * 12,
        local_mixer=[[5, 5], [5, 5], [5, 5]],
        mlp_ratio=4,
        qkv_bias=True,
        out_channels=384,
        out_char_num=40,
        epsilon=1e-6,
        **kwargs)
    return model


def SVTR_large(pretrained=False, use_ssld=False, **kwargs):
    model = SVTRNet(
        img_size=[48, 320],
        embed_dim=[192, 256, 512],
        depth=[6, 6, 9],
        num_heads=[6, 8, 16],
        mixer=['Conv'] * 9 + ['Global'] * 12,
        local_mixer=[[5, 5], [5, 5], [5, 5]],
        mlp_ratio=4,
        qkv_bias=True,
        out_channels=512,
        out_char_num=40,
        epsilon=1e-6,
        **kwargs)
    return model