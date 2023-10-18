import numpy as np
import paddle
import paddle.nn as nn
from ..legendary_models.swin_transformer import SwinTransformer, _load_pretrained, \
    PatchEmbed, BasicLayer, SwinTransformerBlock

MODEL_URLS_SOLIDER = {
    "SwinTransformer_tiny_patch4_window7_224_SOLIDER":
        'https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams',
    "SwinTransformer_small_patch4_window7_224_SOLIDER":
        'https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_small_patch4_window7_224_pretrained.pdparams',
    "SwinTransformer_base_patch4_window7_224_SOLIDER":
        'https://paddleclas.bj.bcebos.com/models/SOLIDER/SwinTransformer_base_patch4_window7_224_pretrained.pdparams'
}

__all__ = list(MODEL_URLS_SOLIDER.keys())


class PatchEmbed_SOLIDER(PatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose([0, 2, 1])  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size


class SwinTransformerBlock_SOLIDER(SwinTransformerBlock):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Layer, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(SwinTransformerBlock_SOLIDER, self).__init__(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.check_condition()

    def check_condition(self):
        if min(self.input_resolution) < self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"


class BasicLayer_SOLIDER(BasicLayer):
    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False):

        super(BasicLayer_SOLIDER, self).__init__(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint
        )
        # build blocks
        self.blocks = nn.LayerList([
            SwinTransformerBlock_SOLIDER(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i]
                if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x_down = self.downsample(x)
            return x_down, x
        else:
            return x, x


class PatchMerging_SOLIDER(nn.Layer):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Layer, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.sampler = nn.Unfold(kernel_sizes=2, strides=2)
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias_attr=False)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, "x size ({}*{}) are not even.".format(
            H, W)

        x = x.reshape([B, H, W, C]).transpose([0, 3, 1, 2])

        x = self.sampler(x)
        x = x.transpose([0, 2, 1])
        x = self.norm(x)
        x = self.reduction(x)
        return x


class SwinTransformer_SOLIDER(SwinTransformer):
    def __init__(self,
                 embed_dim=96,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 class_num=1000,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 out_indices=(0, 1, 2, 3),
                 semantic_weight=1.0,
                 use_checkpoint=False,
                 **kwargs):
        super(SwinTransformer_SOLIDER, self).__init__()
        patches_resolution = self.patch_embed.patches_resolution
        self.num_classes = num_classes = class_num
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        # stochastic depth
        dpr = np.linspace(0, drop_path_rate,
                          sum(depths)).tolist()  # stochastic depth decay rule
        self.patch_embed = PatchEmbed_SOLIDER(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.out_indices = out_indices
        # build layers
        self.layers = nn.LayerList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer_SOLIDER(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging_SOLIDER
                if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features_s = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        for i in out_indices:
            layer = norm_layer(self.num_features_s[i])
            layer_name = f'norm{i}'
            self.add_sublayer(layer_name, layer)
        self.avgpool = nn.AdaptiveAvgPool2D(1)

        # semantic embedding
        self.semantic_weight = semantic_weight
        if self.semantic_weight >= 0:
            self.semantic_embed_w = nn.LayerList()
            self.semantic_embed_b = nn.LayerList()
            for i in range(len(depths)):
                if i >= len(depths) - 1:
                    i = len(depths) - 2
                semantic_embed_w = nn.Linear(2, self.num_features_s[i + 1])
                semantic_embed_b = nn.Linear(2, self.num_features_s[i + 1])
                self._init_weights(semantic_embed_w)
                self._init_weights(semantic_embed_b)
                self.semantic_embed_w.append(semantic_embed_w)
                self.semantic_embed_b.append(semantic_embed_b)
            self.softplus = nn.Softplus()
        self.head = nn.Linear(
            self.num_features,
            num_classes) if self.num_classes > 0 else nn.Identity()

    def forward_features(self, x, semantic_weight=None):
        if self.semantic_weight >= 0 and semantic_weight is None:
            w = paddle.ones((x.shape[0], 1)) * self.semantic_weight
            w = paddle.concat([w, 1 - w], axis=-1)
            semantic_weight = w.cuda()
        x, hw_shape = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        outs = []

        for i, layer in enumerate(self.layers):
            x, out = layer(x)
            if self.semantic_weight >= 0:
                sw = self.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.softplus(sw) + sb
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.reshape([-1, *hw_shape,
                                   self.num_features_s[i]]).transpose([0, 3, 1, 2])
                hw_shape = [item // 2 for item in hw_shape]
                outs.append(out)

        x = self.avgpool(outs[-1])  # B C 1
        x = paddle.flatten(x, 1)

        return x


def SwinTransformer_tiny_patch4_window7_224_SOLIDER(
        pretrained=False,
        **kwargs):
    model = SwinTransformer_SOLIDER(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.2,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.1
        **kwargs)
    _load_pretrained(
        pretrained,
        model=model,
        model_url=MODEL_URLS_SOLIDER["SwinTransformer_tiny_patch4_window7_224_SOLIDER"],
        **kwargs)
    return model


def SwinTransformer_small_patch4_window7_224_SOLIDER(
        pretrained=False,
        **kwargs):
    model = SwinTransformer_SOLIDER(
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        drop_path_rate=0.3,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs)
    _load_pretrained(
        pretrained,
        model=model,
        model_url=MODEL_URLS_SOLIDER["SwinTransformer_small_patch4_window7_224_SOLIDER"],
        **kwargs)
    return model


def SwinTransformer_base_patch4_window7_224_SOLIDER(
        pretrained=False,
        **kwargs):
    model = SwinTransformer_SOLIDER(
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=7,
        drop_path_rate=0.5,  # if imagenet22k or imagenet22kto1k, set drop_path_rate=0.2
        **kwargs)
    _load_pretrained(
        pretrained,
        model=model,
        model_url=MODEL_URLS_SOLIDER["SwinTransformer_base_patch4_window7_224_SOLIDER"],
        **kwargs)
    return model
