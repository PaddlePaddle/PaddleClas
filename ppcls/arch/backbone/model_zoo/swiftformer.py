"""
SwiftFormer - PaddlePaddle Implementation
"""
import collections.abc
import logging
import os
import copy
import paddle
import paddle.nn as nn
import einops
from itertools import repeat

SwiftFormer_width = {
    "XS": [48, 56, 112, 220],
    "S": [48, 64, 168, 224],
    "l1": [48, 96, 192, 384],
    "l3": [64, 128, 320, 512],
}
SwiftFormer_depth = {
    "XS": [3, 3, 6, 4],
    "S": [3, 3, 9, 6],
    "l1": [4, 3, 10, 5],
    "l3": [4, 4, 12, 6],
}

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class DropPath(nn.Layer):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = paddle.rand(shape)
        random_tensor = paddle.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


def _trunc_normal_init(m):
    """Apply truncated normal initialization to weight."""
    paddle.nn.initializer.TruncatedNormal(std=0.02)(m)


def _zeros_init(m):
    """Apply zeros initialization."""
    paddle.nn.initializer.Constant(0)(m)


def _ones_init(m):
    """Apply ones initialization."""
    paddle.nn.initializer.Constant(1.0)(m)


def stem(in_chs, out_chs):
    """
    Stem Layer that is implemented by two layers of conv.
    Output: sequence of layers with final shape of [B, C, H/4, W/4]
    """
    return nn.Sequential(
        nn.Conv2D(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(num_features=out_chs // 2),
        nn.ReLU(),
        nn.Conv2D(out_chs // 2, out_chs, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2D(num_features=out_chs),
        nn.ReLU(),
    )


def to_2tuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


class Embedding(nn.Layer):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(
        self,
        patch_size=16,
        stride=16,
        padding=0,
        in_chans=3,
        embed_dim=768,
        norm_layer=nn.BatchNorm2D,
    ):
        super().__init__()
        patch_size = to_2tuple(2)(patch_size)
        stride = to_2tuple(2)(stride)
        padding = to_2tuple(2)(padding)
        self.proj = nn.Conv2D(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class ConvEncoder(nn.Layer):
    """
    Implementation of ConvEncoder with 3*3 and 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(
        self, dim, hidden_dim=64, kernel_size=3, drop_path=0.0, use_layer_scale=True
    ):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.BatchNorm2D(num_features=dim)
        self.pwconv1 = nn.Conv2D(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(hidden_dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = paddle.create_parameter(
                shape=paddle.ones(dim).unsqueeze(-1).unsqueeze(-1).shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(1.0)
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            _trunc_normal_init(m.weight)
            if m.bias is not None:
                _zeros_init(m.bias)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class Mlp(nn.Layer):
    """
    Implementation of MLP layer with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    Output: tensor with shape [B, C, H, W]
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = nn.BatchNorm2D(num_features=in_features)
        self.fc1 = nn.Conv2D(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2D(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            _trunc_normal_init(m.weight)
            if m.bias is not None:
                _zeros_init(m.bias)

    def forward(self, x):
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class EfficientAdditiveAttnetion(nn.Layer):
    """
    Efficient Additive Attention module for SwiftFormer.
    Input: tensor in shape [B, N, D]
    Output: tensor in shape [B, N, D]
    """

    def __init__(self, in_dims=512, token_dim=256, num_heads=2):
        super().__init__()
        self.to_query = nn.Linear(in_dims, token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, token_dim * num_heads)
        self.w_g = paddle.create_parameter(
            shape=[token_dim * num_heads, 1],
            dtype='float32',
            default_initializer=paddle.nn.initializer.Normal()
        )
        self.scale_factor = token_dim**-0.5
        self.Proj = nn.Linear(token_dim * num_heads, token_dim * num_heads)
        self.final = nn.Linear(token_dim * num_heads, token_dim)

    def forward(self, x):
        #print(f"[DEBUG] attn input: shape={x.shape}, dtype={x.dtype}, min={x.min().item()}, max={x.max().item()}")
        #print(f"[DEBUG] to_query weight: shape={self.to_query.weight.shape}, dtype={self.to_query.weight.dtype}")
        query = self.to_query(x)
        key = self.to_key(x)
        query = nn.functional.normalize(query, axis=-1)
        key = nn.functional.normalize(key, axis=-1)
        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor
        A = nn.functional.normalize(A, axis=1)
        G = paddle.sum(A * query, axis=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])
        out = self.Proj(G * key) + query
        out = self.final(out)
        return out


class SwiftFormerLocalRepresentation(nn.Layer):
    """
    Local Representation module for SwiftFormer that is implemented by 3*3 depth-wise and point-wise convolutions.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(self, dim, kernel_size=3, drop_path=0.0, use_layer_scale=True):
        super().__init__()
        self.dwconv = nn.Conv2D(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )
        self.norm = nn.BatchNorm2D(num_features=dim)
        self.pwconv1 = nn.Conv2D(dim, dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2D(dim, dim, kernel_size=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale = paddle.create_parameter(
                shape=paddle.ones(dim).unsqueeze(-1).unsqueeze(-1).shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(1.0)
            )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2D):
            _trunc_normal_init(m.weight)
            if m.bias is not None:
                _zeros_init(m.bias)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.use_layer_scale:
            x = input + self.drop_path(self.layer_scale * x)
        else:
            x = input + self.drop_path(x)
        return x


class SwiftFormerEncoder(nn.Layer):
    """
    SwiftFormer Encoder Block for SwiftFormer. It consists of (1) Local representation module,
    (2) EfficientAdditiveAttention, and (3) MLP block.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """

    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        drop=0.0,
        drop_path=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-05,
    ):
        super().__init__()
        self.local_representation = SwiftFormerLocalRepresentation(
            dim=dim, kernel_size=3, drop_path=0.0, use_layer_scale=True
        )
        self.attn = EfficientAdditiveAttnetion(in_dims=dim, token_dim=dim, num_heads=1)
        self.linear = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = paddle.create_parameter(
                shape=paddle.ones(dim).unsqueeze(-1).unsqueeze(-1).shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(layer_scale_init_value)
            )
            self.layer_scale_2 = paddle.create_parameter(
                shape=paddle.ones(dim).unsqueeze(-1).unsqueeze(-1).shape,
                dtype='float32',
                default_initializer=paddle.nn.initializer.Constant(layer_scale_init_value)
            )

    def forward(self, x):
        x = self.local_representation(x)
        B, C, H, W = x.shape
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1
                * self.attn(x.transpose([0, 2, 3, 1]).contiguous().reshape([B, H * W, C]))
                .reshape([B, H, W, C])
                .transpose([0, 3, 1, 2])
            )
            x = x + self.drop_path(self.layer_scale_2 * self.linear(x))
        else:
            x = x + self.drop_path(
                self.attn(x.transpose([0, 2, 3, 1]).contiguous().reshape([B, H * W, C]))
                .reshape([B, H, W, C])
                .transpose([0, 3, 1, 2])
            )
            x = x + self.drop_path(self.linear(x))
        return x


def Stage(
    dim,
    index,
    layers,
    mlp_ratio=4.0,
    act_layer=nn.GELU,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-05,
    vit_num=1,
):
    """
    Implementation of each SwiftFormer stages. Here, SwiftFormerEncoder used as the last block
    in all stages, while ConvEncoder used in the rest of the blocks.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H, W]
    """
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = (
            drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        )
        if layers[index] - block_idx <= vit_num:
            blocks.append(
                SwiftFormerEncoder(
                    dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                )
            )
        else:
            blocks.append(
                ConvEncoder(dim=dim, hidden_dim=int(mlp_ratio * dim), kernel_size=3)
            )
    blocks = nn.Sequential(*blocks)
    return blocks


class SwiftFormer(nn.Layer):
    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=4,
        downsamples=None,
        act_layer=nn.GELU,
        num_classes=1000,
        down_patch_size=3,
        down_stride=2,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-05,
        fork_feat=False,
        init_cfg=None,
        pretrained=None,
        vit_num=1,
        distillation=True,
        **kwargs,
    ):
        super().__init__()
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat
        self.patch_embed = stem(3, embed_dims[0])
        network = []
        for i in range(len(layers)):
            stage = Stage(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )
        self.network = nn.LayerList(network)
        if self.fork_feat:
            self.out_indices = [0, 2, 4, 6]
            for i_emb, i_layer in enumerate(self.out_indices):
                if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                    layer = nn.Identity()
                else:
                    layer = nn.BatchNorm2D(num_features=embed_dims[i_emb])
                layer_name = f"norm{i_layer}"
                self.add_sublayer(layer_name, layer)
        else:
            self.norm = nn.BatchNorm2D(num_features=embed_dims[-1])
            self.head = (
                nn.Linear(embed_dims[-1], num_classes)
                if num_classes > 0
                else nn.Identity()
            )
            self.dist = distillation
            if self.dist:
                self.dist_head = (
                    nn.Linear(embed_dims[-1], num_classes)
                    if num_classes > 0
                    else nn.Identity()
                )
        self.apply(self._init_weights)
        self.init_cfg = copy.deepcopy(init_cfg)
        if self.fork_feat and (self.init_cfg is not None or pretrained is not None):
            self.init_weights()

    def init_weights(self, pretrained=None):
        logger = logging.getLogger(__name__)
        if self.init_cfg is None and pretrained is None:
            logger.warning(
                f"No pre-trained weights for {self.__class__.__name__}, training start from scratch"
            )
        else:
            assert (
                "checkpoint" in self.init_cfg
            ), f"Only support specify `Pretrained` in `init_cfg` in {self.__class__.__name__} "
            if self.init_cfg is not None:
                ckpt_path = self.init_cfg["checkpoint"]
            elif pretrained is not None:
                ckpt_path = pretrained
            logger.info(f"Loading pretrained weights from {ckpt_path}")
            ckpt = paddle.load(ckpt_path)
            if "state_dict" in ckpt:
                _state_dict = ckpt["state_dict"]
            elif "model" in ckpt:
                _state_dict = ckpt["model"]
            else:
                _state_dict = ckpt
            missing_keys, unexpected_keys = self.set_state_dict(_state_dict)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            _trunc_normal_init(m.weight)
            if m.bias is not None:
                _zeros_init(m.bias)
        elif isinstance(m, nn.LayerNorm):
            _zeros_init(m.bias)
            _ones_init(m.weight)

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            if self.fork_feat and idx in self.out_indices:
                norm_layer = getattr(self, f"norm{idx}")
                x_out = norm_layer(x)
                outs.append(x_out)
        if self.fork_feat:
            return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            return x
        x = self.norm(x)
        x = x.flatten(2).mean(-1)
        if self.dist:
            cls_logits = self.head(x)
            dist_logits = self.dist_head(x)
            if self.training:
                # Keep dual-head outputs while exposing a default logits tensor for existing losses.
                return {
                    "logits": (cls_logits + dist_logits) / 2,
                    "cls_logits": cls_logits,
                    "dist_logits": dist_logits,
                }
            return (cls_logits + dist_logits) / 2
        return self.head(x)


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


def SwiftFormer_XS(pretrained=False, **kwargs):
    model = SwiftFormer(
        layers=SwiftFormer_depth["XS"],
        embed_dims=SwiftFormer_width["XS"],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        model.init_weights()
    return model


def SwiftFormer_S(pretrained=False, **kwargs):
    model = SwiftFormer(
        layers=SwiftFormer_depth["S"],
        embed_dims=SwiftFormer_width["S"],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        model.init_weights()
    return model


def SwiftFormer_L1(pretrained=False, **kwargs):
    model = SwiftFormer(
        layers=SwiftFormer_depth["l1"],
        embed_dims=SwiftFormer_width["l1"],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        model.init_weights()
    return model


def SwiftFormer_L3(pretrained=False, **kwargs):
    model = SwiftFormer(
        layers=SwiftFormer_depth["l3"],
        embed_dims=SwiftFormer_width["l3"],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs,
    )
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        model.init_weights()
    return model
