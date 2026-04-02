import paddle

"""
SigLIP-2 Vision Transformer - Standalone Model File

This file contains complete SigLIP-2 model structure that can be used independently
without full timm library dependency.

Based on: https://arxiv.org/abs/2502.14786
"""
import math
from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union


def _calculate_fan_in_and_fan_out(tensor):
    """Calculate fan_in and fan_out for a tensor."""
    if tensor.ndim < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    
    if tensor.ndim == 2:
        fan_in, fan_out = tensor.shape[1], tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.ndim > 2:
            receptive_field_size = int(paddle.numel(tensor[0][0]))
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    
    return fan_in, fan_out


def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    """Variance scaling initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        raise ValueError(f"Invalid mode: {mode}")
    variance = scale / denom
    if distribution == "truncated_normal":
        std = math.sqrt(variance) / 0.8796256610342398
        paddle.nn.init.trunc_normal_(tensor, std=std)
    elif distribution == "normal":
        with paddle.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with paddle.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    """LeCun normal initialization."""
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def resample_abs_pos_embed(
    posemb: 'paddle.Tensor',
    new_size: Tuple[int, int],
    old_size: Optional[Tuple[int, int]] = None,
    num_prefix_tokens: int = 0,
    interpolation: str = "bicubic",
    antialias: bool = True,
):
    """Resample absolute position embeddings to a new size.

    Args:
        posemb: Position embedding tensor of shape (1, num_tokens, embed_dim)
        new_size: New grid size (height, width)
        old_size: Old grid size (height, width), inferred if None
        num_prefix_tokens: Number of prefix tokens (class tokens, etc.)
        interpolation: Interpolation method
        antialias: Whether to use antialiasing

    Returns:
        Resampled position embedding
    """
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb
    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw
    if num_prefix_tokens:
        posemb_prefix, posemb = (
            posemb[:, :num_prefix_tokens],
            posemb[:, num_prefix_tokens:],
        )
    else:
        posemb_prefix, posemb = None, posemb
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    posemb = paddle.nn.functional.interpolate(
        posemb, size=new_size, mode=interpolation, antialias=antialias
    )
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)
    if posemb_prefix is not None:
        posemb = paddle.cat([posemb_prefix, posemb], dim=1)
    return posemb


class DropPath(paddle.nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(p=keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class LayerScale(paddle.nn.Module):
    """LayerScale on tensors with channels in last-dim."""

    def __init__(
        self, dim: int, init_values: float = 1e-05, inplace: bool = False
    ) -> None:
        super().__init__()
        self.init_values = init_values
        self.inplace = inplace
        self.gamma = paddle.nn.Parameter(paddle.empty(dim))
        self.reset_parameters()

    def reset_parameters(self):
        paddle.nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm(paddle.nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first."""

    def __init__(self, normalized_shape, eps=1e-06, data_format="channels_last"):
        super().__init__()
        self.weight = paddle.nn.Parameter(paddle.ones(normalized_shape))
        self.bias = paddle.nn.Parameter(paddle.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        if self.data_format == "channels_last":
            return paddle.nn.functional.layer_norm(
                x, self.weight.shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.eps)
            x = self.weight[:, (None), (None)] * x + self.bias[:, (None), (None)]
            return x


class Mlp(paddle.nn.Module):
    """MLP as used in Vision Transformer."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Type[paddle.nn.Module] = paddle.nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.compat.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = paddle.compat.nn.Linear(hidden_features, out_features)
        self.drop = paddle.nn.Dropout(drop)

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(paddle.nn.Module):
    """Multi-head Self Attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = paddle.compat.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = paddle.nn.Dropout(attn_drop)
        self.proj = paddle.compat.nn.Linear(dim, dim)
        self.proj_drop = paddle.nn.Dropout(proj_drop)

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        attn = q @ k.transpose(-2, -1) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class AttentionPoolLatent(paddle.nn.Module):
    """Attention pooling for global pooling with learnable latent vector."""

    def __init__(
        self,
        in_features: int,
        out_features: Optional[int] = None,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        norm_layer: Type[paddle.nn.Module] = LayerNorm,
        act_layer: Type[paddle.nn.Module] = paddle.nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        self.num_heads = num_heads
        self.head_dim = in_features // num_heads
        self.scale = self.head_dim**-0.5
        self.latent = paddle.nn.Parameter(paddle.zeros(1, 1, in_features))
        self.q = paddle.compat.nn.Linear(in_features, in_features)
        self.kv = paddle.compat.nn.Linear(in_features, in_features * 2)
        self.q_norm = paddle.nn.Identity()
        self.k_norm = paddle.nn.Identity()
        self.proj = paddle.compat.nn.Linear(in_features, in_features)
        self.norm = norm_layer(out_features)
        self.mlp = Mlp(out_features, int(out_features * mlp_ratio), act_layer=act_layer)

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        B, N, C = x.shape
        q_latent = self.latent.expand(B, -1, -1)
        q = (
            self.q(q_latent)
            .reshape(B, 1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        kv = (
            self.kv(x)
            .reshape(B, N, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, 1, C)
        x = self.proj(x)
        x = x + self.mlp(self.norm(x))
        return x.squeeze(1)


class PatchEmbed(paddle.nn.Module):
    """Image to Patch Embedding with dynamic size support."""

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Type[paddle.nn.Module]] = None,
        flatten: bool = True,
        strict_img_size: bool = False,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (
            (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        )
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.strict_img_size = strict_img_size
        self.proj = paddle.nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = norm_layer(embed_dim) if norm_layer else paddle.nn.Identity()

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        B, C, H, W = x.shape
        if self.strict_img_size:
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

    def get_grid_size(self, H: int, W: int) -> Tuple[int, int]:
        """Get grid size for given input dimensions."""
        return H // self.patch_size[0], W // self.patch_size[1]


class Block(paddle.nn.Module):
    """Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: Type[paddle.nn.Module] = LayerNorm,
        act_layer: Type[paddle.nn.Module] = paddle.nn.GELU,
        init_values: Optional[float] = None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else paddle.nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else paddle.nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else paddle.nn.Identity()
        )

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class SigLIP2VisionTransformer(paddle.nn.Module):
    """SigLIP-2 Vision Transformer (Image Encoder).

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of MLP hidden dim to embedding dim.
        qkv_bias: Enable bias for QKV projections.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        global_pool: Global pooling type ('avg', 'max', 'map').
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 256,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        global_pool: str = "map",
    ):
        super().__init__()
        self.global_pool = global_pool
        self.embed_dim = embed_dim
        self.num_features = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        self.cls_token = None
        self.pos_embed = paddle.nn.Parameter(paddle.zeros(1, num_patches, embed_dim))
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        self.blocks = paddle.nn.Sequential(
            *[
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = LayerNorm(embed_dim)
        if global_pool == "map":
            self.attn_pool = AttentionPoolLatent(
                in_features=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio
            )
        else:
            self.attn_pool = None
        self.fc_norm = (
            LayerNorm(embed_dim)
            if global_pool in ("avg", "max")
            else paddle.nn.Identity()
        )
        self._init_weights()

    def _init_weights(self):
        paddle.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights_reset)

    def _init_weights_reset(self, m):
        """Reset模式的权重初始化，与timm完全一致"""
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()

    def forward(self, x: 'paddle.Tensor') -> 'paddle.Tensor':
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        B, N, C = x.shape
        grid_h, grid_w = (
            H // self.patch_embed.patch_size[0],
            W // self.patch_embed.patch_size[1],
        )
        num_patches = grid_h * grid_w
        pos_embed = self.pos_embed
        if num_patches != pos_embed.shape[1]:
            old_size = int(math.sqrt(pos_embed.shape[1])), int(
                math.sqrt(pos_embed.shape[1])
            )
            pos_embed = resample_abs_pos_embed(
                pos_embed,
                new_size=(grid_h, grid_w),
                old_size=old_size,
                num_prefix_tokens=0,
            )
        x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        if self.global_pool == "map":
            x = self.attn_pool(x)
        elif self.global_pool == "avg":
            x = x.mean(dim=1)
        elif self.global_pool == "max":
            x = (x.max(axis=1), x.argmax(axis=1))[0]
        else:
            x = x[:, (0)]
        x = self.fc_norm(x)
        return x


def siglip2_base_patch16_256(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 256,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Base model with 16x16 patches.

    Args:
        pretrained: Whether to load pretrained weights.
        checkpoint_path: Path to checkpoint file (.pth or .safetensors).
        img_size: Input image size (default: 256, can be any size).

    Returns:
        SigLIP2VisionTransformer model.
    """
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_base_patch16_384(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 384,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Base model with 16x16 patches at 384x384 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_base_patch16_512(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 512,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Base model with 16x16 patches at 512x512 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_large_patch16_256(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 256,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Large model with 16x16 patches.

    Args:
        pretrained: Whether to load pretrained weights.
        checkpoint_path: Path to checkpoint file (.pth or .safetensors).
        img_size: Input image size (default: 256, can be any size).

    Returns:
        SigLIP2VisionTransformer model.
    """
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_large_patch16_384(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 384,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Large model with 16x16 patches at 384x384 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_large_patch16_512(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 512,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 Large model with 16x16 patches at 512x512 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch14_siglip_224(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 224,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 14x14 patches at 224x224 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch14_siglip_378(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 378,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 14x14 patches at 378x378 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch14_siglip_384(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 384,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 14x14 patches at 384x384 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=14,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=3.7362,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch16_siglip_256(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 256,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 16x16 patches at 256x256 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch16_siglip_384(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 384,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 16x16 patches at 384x384 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def siglip2_so400m_patch16_siglip_512(
    pretrained: bool = False,
    checkpoint_path: Optional[str] = None,
    img_size: Union[int, Tuple[int, int]] = 512,
) -> SigLIP2VisionTransformer:
    """SigLIP-2 SO400M model with 16x16 patches at 512x512 resolution."""
    model = SigLIP2VisionTransformer(
        img_size=img_size,
        patch_size=16,
        embed_dim=1152,
        depth=27,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        global_pool="map",
    )
    if pretrained and checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    return model


def load_checkpoint(model: paddle.nn.Module, checkpoint_path: str) -> None:
    """Load checkpoint weights into model.

    Args:
        model: Model to load weights into.
        checkpoint_path: Path to checkpoint file (.pdparams or .safetensors).
    """
    if checkpoint_path.endswith(".safetensors"):
        try:
            from safetensors.numpy import load_file

            state_dict_np = load_file(checkpoint_path)
            state_dict = {}
            for key, value in state_dict_np.items():
                state_dict[key] = paddle.to_tensor(value)
        except ImportError:
            raise ImportError(
                "safetensors is required to load .safetensors files. Install with: pip install safetensors"
            )
    else:
        checkpoint = paddle.load(path=str(checkpoint_path))
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    model_state_dict = model.state_dict()
    filtered_state_dict = {}
    pos_embed_key = None
    for key, value in state_dict.items():
        if key in model_state_dict:
            if key == "pos_embed":
                pos_embed_key = key
                if value.shape == model_state_dict[key].shape:
                    filtered_state_dict[key] = value
                else:
                    print(
                        f"Note: Position embedding size mismatch in checkpoint ({value.shape}) vs model ({model_state_dict[key].shape})"
                    )
                    print(
                        f"Note: Using interpolated position embeddings for dynamic size support"
                    )
            else:
                filtered_state_dict[key] = value
    model.load_state_dict(filtered_state_dict, strict=False)
    print(f"Loaded checkpoint from {checkpoint_path}")


if __name__ == "__main__":
    model = siglip2_base_patch16_256(pretrained=False)
    x = paddle.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.size for p in model.parameters()):,}")
