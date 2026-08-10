import sys
import os
import warnings
import paddle

sys.path.insert(0, ".")
sys.path.insert(0, "..")
from paddle_utils import *

warnings.filterwarnings("ignore")

# 切换到脚本所在目录，保证 ./weights、./imagenet_official、./video_test_data 等相对路径可用。
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from PIL import Image
from tqdm import tqdm
import inspect
import math
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


# Embedded Hiera implementation (merged from hiera/hfhub.py, hiera/hiera_utils.py, hiera/hiera.py).
class PyTorchModelHubMixin:
    error_str: str = (
        'This feature requires "huggingface-hub >= 0.21.0" to be installed.'
    )

    @classmethod
    def from_pretrained(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)

    @classmethod
    def save_pretrained(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)

    @classmethod
    def push_to_hub(cls, *args, **kwdargs):
        raise RuntimeError(cls.error_str)


def has_config(func):
    signature = inspect.signature(func)

    def wrapper(self, *args, **kwdargs):
        if "config" in kwdargs:
            config = kwdargs["config"]
            del kwdargs["config"]
            kwdargs.update(**config)
        self.config = {
            k: (v.default if i - 1 >= len(args) else args[i - 1])
            for i, (k, v) in enumerate(signature.parameters.items())
            if v.default is not inspect.Parameter.empty
        }
        self.config.update(**kwdargs)
        func(self, **kwdargs)

    return wrapper


PADDLE_WEIGHTS_DIR = "./weights"

PADDLE_WEIGHTS_MAP = {
    "hiera_tiny_224": {
        "mae_in1k_ft_in1k": "hiera_tiny_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_tiny_224_mae_in1k.pdparams",
    },
    "hiera_small_224": {
        "mae_in1k_ft_in1k": "hiera_small_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_small_224_mae_in1k.pdparams",
    },
    "hiera_base_224": {
        "mae_in1k_ft_in1k": "hiera_base_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_base_224_mae_in1k.pdparams",
    },
    "hiera_base_plus_224": {
        "mae_in1k_ft_in1k": "hiera_base_plus_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_base_plus_224_mae_in1k.pdparams",
    },
    "hiera_large_224": {
        "mae_in1k_ft_in1k": "hiera_large_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_large_224_mae_in1k.pdparams",
    },
    "hiera_huge_224": {
        "mae_in1k_ft_in1k": "hiera_huge_224_mae_in1k_ft_in1k.pdparams",
        "mae_in1k": "hiera_huge_224_mae_in1k.pdparams",
    },
    "hiera_base_16x224": {
        "mae_k400_ft_k400": "hiera_base_16x224_mae_k400_ft_k400.pdparams",
        "mae_k400": "hiera_base_16x224_mae_k400.pdparams",
    },
    "hiera_base_plus_16x224": {
        "mae_k400_ft_k400": "hiera_base_plus_16x224_mae_k400_ft_k400.pdparams",
        "mae_k400": "hiera_base_plus_16x224_mae_k400.pdparams",
    },
    "hiera_large_16x224": {
        "mae_k400_ft_k400": "hiera_large_16x224_mae_k400_ft_k400.pdparams",
        "mae_k400": "hiera_large_16x224_mae_k400.pdparams",
    },
    "hiera_huge_16x224": {
        "mae_k400_ft_k400": "hiera_huge_16x224_mae_k400_ft_k400.pdparams",
        "mae_k400": "hiera_huge_16x224_mae_k400.pdparams",
    },
}


def load_paddle_weights(model_name: str, checkpoint: str) -> Dict:
    """Load Paddle weights from local file."""
    if model_name not in PADDLE_WEIGHTS_MAP:
        return None
    
    checkpoint_map = PADDLE_WEIGHTS_MAP[model_name]
    if checkpoint not in checkpoint_map:
        return None
    
    weight_file = checkpoint_map[checkpoint]
    weight_path = os.path.join(PADDLE_WEIGHTS_DIR, weight_file)
    
    if os.path.exists(weight_path):
        print(f"Loading Paddle weights from: {weight_path}")
        return paddle.load(weight_path)
    
    raise FileNotFoundError(f"Paddle weights not found: {weight_path}")


def load_pytorch_weights(url: str) -> Dict:
    """Load PyTorch weights and convert to PaddlePaddle format."""
    try:
        import torch
        import urllib.request
    except ImportError:
        raise ImportError("PyTorch is required to load PyTorch weights. Please install it with: pip install torch")
    
    cache_dir = os.path.expanduser("~/.cache/paddle/hub/checkpoints")
    os.makedirs(cache_dir, exist_ok=True)
    
    filename = url.split("/")[-1]
    cached_file = os.path.join(cache_dir, filename)
    
    if not os.path.exists(cached_file):
        print(f"Downloading: {url} to {cached_file}")
        urllib.request.urlretrieve(url, cached_file)
    
    state_dict = torch.load(cached_file, map_location="cpu", weights_only=False)
    if isinstance(state_dict, dict):
        if "model_state" in state_dict:
            state_dict = state_dict["model_state"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    
    paddle_state_dict = {}
    for key, value in state_dict.items():
        if hasattr(value, "numpy"):
            tensor_value = paddle.to_tensor(value.detach().numpy())
        elif isinstance(value, np.ndarray):
            tensor_value = paddle.to_tensor(value)
        else:
            tensor_value = value
        
        if "mlp.fc" in key and key.endswith(".weight") and len(tensor_value.shape) == 2:
            tensor_value = tensor_value.T
        
        paddle_state_dict[key] = tensor_value
    
    return paddle_state_dict


def pretrained_model(checkpoints: Dict[str, str], default: str = None) -> Callable:
    """Loads a Hiera model from a pretrained source (if pretrained=True). Use "checkpoint" to specify the checkpoint."""

    def inner(model_func: Callable) -> Callable:
        model_name = model_func.__name__
        
        def model_def(
            pretrained: bool = False,
            checkpoint: str = default,
            strict: bool = True,
            **kwdargs,
        ) -> paddle.nn.Module:
            if pretrained:
                if checkpoints is None:
                    raise RuntimeError(
                        "This model currently doesn't have pretrained weights available."
                    )
                elif checkpoint is None:
                    raise RuntimeError("No checkpoint specified.")
                elif checkpoint not in checkpoints:
                    raise RuntimeError(
                        f"Invalid checkpoint specified ({checkpoint}). Options are: {list(checkpoints.keys())}."
                    )
                
                state_dict = load_paddle_weights(model_name, checkpoint)
                
                if state_dict is not None and "head.projection.weight" in state_dict:
                    head_weight = state_dict["head.projection.weight"]
                    num_classes_from_ckpt = head_weight.shape[0]
                    if "num_classes" not in kwdargs:
                        kwdargs["num_classes"] = num_classes_from_ckpt
                    elif kwdargs["num_classes"] != num_classes_from_ckpt:
                        del state_dict["head.projection.weight"]
                        del state_dict["head.projection.bias"]
            model = model_func(**kwdargs)
            if pretrained and state_dict is not None:
                if "decoder_pos_embed" in state_dict and not hasattr(
                    model, "decoder_pos_embed"
                ):
                    strict = False
                model.set_state_dict(state_dict)
            return model

        model_def.checkpoints = checkpoints
        model_def.default = default
        return model_def

    return inner


def conv_nd(n: int) -> Type[paddle.nn.Module]:
    return [paddle.nn.Identity, paddle.nn.Conv1d, paddle.nn.Conv2d, paddle.nn.Conv3d][n]


def do_pool(x: paddle.Tensor, stride: int) -> paddle.Tensor:
    return x.view(x.shape[0], stride, -1, x.shape[-1]).max(axis=1)


def get_resized_mask(target_size: paddle.Size, mask: paddle.Tensor) -> paddle.Tensor:
    if mask is None:
        return mask
    assert len(mask.shape[2:]) == len(target_size)
    if mask.shape[2:] != target_size:
        return paddle.nn.functional.interpolate(mask.float(), size=target_size)
    return mask


def do_masked_conv(
    x: paddle.Tensor, conv: paddle.nn.Module, mask: Optional[paddle.Tensor] = None
) -> paddle.Tensor:
    if conv is None:
        return x
    if mask is None:
        return conv(x)
    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
    x: paddle.Tensor, shape: List[int], mu_shape: List[int]
) -> paddle.Tensor:
    D = len(shape)
    B, C = x.shape[0], x.shape[-1]
    num_MUs = [(s // mu) for s, mu in zip(shape, mu_shape)]
    x = x.view(B, *num_MUs, *mu_shape, C)
    permute = (
        [0]
        + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D, 1 + 2 * D))], [])
        + [len(x.shape) - 1]
    )
    x = x.permute(permute).reshape(B, *shape, C)
    return x


class Unroll(paddle.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
    ):
        super().__init__()
        self.size = [(i // s) for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        B, _, C = x.shape
        cur_size = self.size
        x = x.view(*([B] + cur_size + [C]))
        for strides in self.schedule:
            cur_size = [(i // s) for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.view(new_shape)
            L = len(new_shape)
            permute = (
                [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            )
            x = x.permute(permute)
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)
        x = x.reshape(-1, math.prod(self.size), C)
        return x


class Reroll(paddle.nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        patch_stride: Tuple[int, ...],
        unroll_schedule: List[Tuple[int, ...]],
        stage_ends: List[int],
        q_pool: int,
    ):
        super().__init__()
        self.size = [(i // s) for i, s in zip(input_size, patch_stride)]
        self.schedule = {}
        size = self.size
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = unroll_schedule, size
            if i in stage_ends[:q_pool]:
                if len(unroll_schedule) > 0:
                    size = [(n // s) for n, s in zip(size, unroll_schedule[0])]
                unroll_schedule = unroll_schedule[1:]

    def forward(
        self, x: paddle.Tensor, block_idx: int, mask: paddle.Tensor = None
    ) -> paddle.Tensor:
        schedule, size = self.schedule[block_idx]
        B, N, C = x.shape
        D = len(size)
        cur_mu_shape = [1] * D
        for strides in schedule:
            x = x.view(B, *strides, N // math.prod(strides), *cur_mu_shape, C)
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum(
                    [list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))], []
                )
                + [L - 1]
            )
            x = x.permute(permute)
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]
        x = x.view(B, N, *cur_mu_shape, C)
        if mask is not None:
            return x
        x = undo_windowing(x, size, cur_mu_shape)
        return x


class Mlp(paddle.nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        drop: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = paddle.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = paddle.nn.Linear(hidden_features, out_features)
        self.drop = paddle.nn.Dropout(drop)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(paddle.nn.Layer):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = paddle.bernoulli(paddle.full(shape, keep_prob, dtype=x.dtype))
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor = random_tensor / keep_prob
        return x * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class MaskUnitAttention(paddle.nn.Module):
    """
    Computes either Mask Unit or Global Attention. Also is able to perform q pooling.

    Note: this assumes the tokens have already been flattened and unrolled into mask units.
    See `Unroll` for more details.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        """
        Args:
        - dim, dim_out: The input and output feature dimensions.
        - heads: The number of attention heads.
        - q_stride: If greater than 1, pool q with this stride. The stride should be flattened (e.g., 2x2 = 4).
        - window_size: The current (flattened) size of a mask unit *after* pooling (if any).
        - use_mask_unit_attn: Use Mask Unit or Global Attention.
        """
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim**-0.5
        self.qkv = paddle.compat.nn.Linear(dim, 3 * dim_out)
        self.proj = paddle.compat.nn.Linear(dim_out, dim_out)
        self.window_size = window_size
        self.use_mask_unit_attn = use_mask_unit_attn

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """Input should be of shape [batch, tokens, channels]."""
        B, N, _ = x.shape
        num_windows = (
            N // (self.q_stride * self.window_size) if self.use_mask_unit_attn else 1
        )
        qkv = (
            self.qkv(x)
            .reshape(B, -1, num_windows, 3, self.heads, self.head_dim)
            .permute(3, 0, 4, 2, 1, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.q_stride > 1:
            q = q.view(
                B, self.heads, num_windows, self.q_stride, -1, self.head_dim
            ).max(axis=3)
        attn = q * self.scale @ k.transpose(-1, -2)
        attn = paddle.nn.functional.softmax(attn, axis=-1)
        x = attn @ v
        x = x.transpose(1, 3).reshape(B, -1, self.dim_out)
        x = self.proj(x)
        return x


class HieraBlock(paddle.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: int,
        heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        norm_layer: paddle.nn.Module = paddle.nn.LayerNorm,
        act_layer: paddle.nn.Module = paddle.nn.GELU,
        q_stride: int = 1,
        window_size: int = 0,
        use_mask_unit_attn: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.attn = MaskUnitAttention(
            dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn
        )
        self.norm2 = norm_layer(dim_out)
        self.mlp = Mlp(
            dim_out, int(dim_out * mlp_ratio), act_layer=act_layer
        )
        self.drop_path = (
            DropPath(drop_path)
            if drop_path > 0
            else paddle.nn.Identity()
        )
        if dim != dim_out:
            self.proj = paddle.compat.nn.Linear(dim, dim_out)

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x_norm = self.norm1(x)
        if self.dim != self.dim_out:
            x = do_pool(self.proj(x_norm), stride=self.attn.q_stride)
        x = x + self.drop_path(self.attn(x_norm))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(paddle.nn.Module):
    def __init__(
        self,
        dim: int,
        num_classes: int,
        dropout_rate: float = 0.0,
        act_func: Callable[[paddle.Tensor], paddle.Tensor] = lambda x: paddle.nn.functional.softmax(
            x, axis=-1
        ),
    ):
        super().__init__()
        self.dropout = (
            paddle.nn.Dropout(dropout_rate)
            if dropout_rate > 0
            else paddle.nn.Identity()
        )
        self.projection = paddle.compat.nn.Linear(dim, num_classes)
        self.act_func = act_func

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        x = self.dropout(x)
        x = self.projection(x)
        if not self.training:
            x = self.act_func(x)
        return x


class PatchEmbed(paddle.nn.Module):
    """Patch embed that supports any number of spatial dimensions (1d, 2d, 3d)."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        kernel: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
    ):
        super().__init__()
        self.spatial_dims = len(kernel)
        self.proj = conv_nd(self.spatial_dims)(
            dim_in, dim_out, kernel_size=kernel, stride=stride, padding=padding
        )

    def forward(
        self, x: paddle.Tensor, mask: Optional[paddle.Tensor] = None
    ) -> paddle.Tensor:
        x = do_masked_conv(x, self.proj, mask)
        x = x.reshape(x.shape[0], x.shape[1], -1).transpose(2, 1)
        return x


class Hiera(paddle.nn.Module, PyTorchModelHubMixin):
    @has_config
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        num_classes: int = 1000,
        stages: Tuple[int, ...] = (2, 3, 16, 3),
        q_pool: int = 3,
        q_stride: Tuple[int, ...] = (2, 2),
        mask_unit_size: Tuple[int, ...] = (8, 8),
        mask_unit_attn: Tuple[bool, ...] = (True, True, False, False),
        dim_mul: float = 2.0,
        head_mul: float = 2.0,
        patch_kernel: Tuple[int, ...] = (7, 7),
        patch_stride: Tuple[int, ...] = (4, 4),
        patch_padding: Tuple[int, ...] = (3, 3),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.0,
        norm_layer: Union[str, paddle.nn.Module] = "LayerNorm",
        head_dropout: float = 0.0,
        head_init_scale: float = 0.001,
        sep_pos_embed: bool = False,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(paddle.nn, norm_layer), eps=1e-06)
        depth = sum(stages)
        self.patch_stride = patch_stride
        self.tokens_spatial_shape = [(i // s) for i, s in zip(input_size, patch_stride)]
        num_tokens = math.prod(self.tokens_spatial_shape)
        flat_mu_size = math.prod(mask_unit_size)
        flat_q_stride = math.prod(q_stride)
        assert q_pool < len(stages)
        self.q_pool, self.q_stride = q_pool, q_stride
        self.mu_size, self.mask_unit_size = flat_mu_size, mask_unit_size
        self.mask_spatial_shape = [
            (i // s) for i, s in zip(self.tokens_spatial_shape, self.mask_unit_size)
        ]
        self.stage_ends = [(sum(stages[:i]) - 1) for i in range(1, len(stages) + 1)]
        self.patch_embed = PatchEmbed(
            in_chans, embed_dim, patch_kernel, patch_stride, patch_padding
        )
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            self.pos_embed_spatial = paddle.nn.Parameter(
                paddle.zeros(
                    1,
                    self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                    embed_dim,
                )
            )
            self.pos_embed_temporal = paddle.nn.Parameter(
                paddle.zeros(1, self.tokens_spatial_shape[0], embed_dim)
            )
        else:
            self.pos_embed = paddle.nn.Parameter(paddle.zeros(1, num_tokens, embed_dim))
        self.unroll = Unroll(
            input_size, patch_stride, [q_stride] * len(self.stage_ends[:-1])
        )
        self.reroll = Reroll(
            input_size,
            patch_stride,
            [q_stride] * len(self.stage_ends[:-1]),
            self.stage_ends,
            q_pool,
        )
        q_pool_blocks = [(x + 1) for x in self.stage_ends[:q_pool]]
        dpr = [x.item() for x in paddle.linspace(0, drop_path_rate, depth)]
        cur_stage = 0
        self.blocks = paddle.nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            use_mask_unit_attn = mask_unit_attn[cur_stage]
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
                if i in q_pool_blocks:
                    flat_mu_size //= flat_q_stride
            block = HieraBlock(
                dim=embed_dim,
                dim_out=dim_out,
                heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                q_stride=flat_q_stride if i in q_pool_blocks else 1,
                window_size=flat_mu_size,
                use_mask_unit_attn=use_mask_unit_attn,
            )
            embed_dim = dim_out
            self.blocks.append(block)
        self.norm = norm_layer(embed_dim)
        self.head = Head(embed_dim, num_classes, dropout_rate=head_dropout)
        if sep_pos_embed:
            paddle.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            paddle.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)
        else:
            paddle.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(partial(self._init_weights))
        self.head.projection.weight.set_value(
            self.head.projection.weight.numpy() * head_init_scale
        )
        self.head.projection.bias.set_value(
            self.head.projection.bias.numpy() * head_init_scale
        )

    def _init_weights(self, m, init_bias=0.02):
        if isinstance(
            m,
            (
                paddle.compat.nn.Linear,
                paddle.nn.Conv1d,
                paddle.nn.Conv2d,
                paddle.nn.Conv3d,
            ),
        ):
            paddle.nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, paddle.compat.nn.Linear) and m.bias is not None:
                paddle.nn.init.constant_(m.bias, init_bias)
        elif isinstance(m, paddle.nn.LayerNorm):
            paddle.nn.init.constant_(m.bias, init_bias)
            paddle.nn.init.constant_(m.weight, 1.0)

    @paddle.jit.not_to_static
    def no_weight_decay(self):
        if self.sep_pos_embed:
            return ["pos_embed_spatial", "pos_embed_temporal"]
        else:
            return ["pos_embed"]

    def get_random_mask(self, x: paddle.Tensor, mask_ratio: float) -> paddle.Tensor:
        """
        Generates a random mask, mask_ratio fraction are dropped.
        1 is *keep*, 0 is *remove*. Useful for MAE, FLIP, etc.
        """
        B = x.shape[0]
        num_windows = math.prod(self.mask_spatial_shape)
        len_keep = int(num_windows * (1 - mask_ratio))
        noise = paddle.rand([B, num_windows])
        ids_shuffle = paddle.argsort(noise, axis=1)
        ids_restore = paddle.argsort(ids_shuffle, axis=1)
        mask = paddle.zeros([B, num_windows])
        mask[:, :len_keep] = 1
        mask = paddle.gather(mask, axis=1, index=ids_restore)
        return mask.bool()

    def get_pos_embed(self) -> paddle.Tensor:
        if self.sep_pos_embed:
            return self.pos_embed_spatial.repeat(
                1, self.tokens_spatial_shape[0], 1
            ) + paddle.repeat_interleave(
                self.pos_embed_temporal,
                self.tokens_spatial_shape[1] * self.tokens_spatial_shape[2],
                axis=1,
            )
        else:
            return self.pos_embed

    def forward(
        self,
        x: paddle.Tensor,
        mask: paddle.Tensor = None,
        return_intermediates: bool = False,
    ) -> paddle.Tensor:
        """
        mask should be a boolean tensor of shape [B, #MUt*#MUy*#MUx] where #MU are the number of mask units in that dim.
        Note: 1 in mask is *keep*, 0 is *remove*; mask.sum(dim=-1) should be the same across the batch.
        """
        if isinstance(x, list):
            x = x[0]
        intermediates = []
        x = self.patch_embed(
            x,
            mask=mask.view(x.shape[0], 1, *self.mask_spatial_shape)
            if mask is not None
            else None,
        )
        x = x + self.get_pos_embed()
        x = self.unroll(x)
        if mask is not None:
            x = x[
                mask[..., None].tile(repeat_times=[1, self.mu_size, x.shape[2]])
            ].view(x.shape[0], -1, x.shape[-1])
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if return_intermediates and i in self.stage_ends:
                intermediates.append(self.reroll(x, i, mask=mask))
        if mask is None:
            x = x.mean(axis=1)
            x = self.norm(x)
            x = self.head(x)
        if return_intermediates:
            return x, intermediates
        return x


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_tiny_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_tiny_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_tiny_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_small_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_small_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_small_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_plus_224(**kwdargs):
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_large_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_large_224(**kwdargs):
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_224.pth",
        "mae_in1k": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_224.pth",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_huge_224(**kwdargs):
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_16x224(num_classes: int = 400, **kwdargs):
    return Hiera(
        num_classes=num_classes,
        input_size=(16, 224, 224),
        q_stride=(1, 2, 2),
        mask_unit_size=(1, 8, 8),
        patch_kernel=(3, 7, 7),
        patch_stride=(2, 4, 4),
        patch_padding=(1, 3, 3),
        sep_pos_embed=True,
        **kwdargs
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_base_plus_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_base_plus_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_large_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_large_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://dl.fbaipublicfiles.com/hiera/hiera_huge_16x224.pth",
        "mae_k400": "https://dl.fbaipublicfiles.com/hiera/mae_hiera_huge_16x224.pth",
    },
    default="mae_k400_ft_k400",
)
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )


MODEL_BUILDERS = {
    "hiera_tiny_224": hiera_tiny_224,
    "hiera_small_224": hiera_small_224,
    "hiera_base_224": hiera_base_224,
    "hiera_base_plus_224": hiera_base_plus_224,
    "hiera_large_224": hiera_large_224,
    "hiera_huge_224": hiera_huge_224,
    "hiera_base_16x224": hiera_base_16x224,
    "hiera_base_plus_16x224": hiera_base_plus_16x224,
    "hiera_large_16x224": hiera_large_16x224,
    "hiera_huge_16x224": hiera_huge_16x224,
}


MODEL_CONFIGS = {'image': {'hiera_tiny_224': {'target_acc': 82.8},
           'hiera_small_224': {'target_acc': 83.8},
           'hiera_base_224': {'target_acc': 84.5},
           'hiera_base_plus_224': {'target_acc': 85.2},
           'hiera_large_224': {'target_acc': 86.1},
           'hiera_huge_224': {'target_acc': 86.9}},
 'video': {'hiera_base_16x224': {'target_acc': 84.0},
           'hiera_base_plus_16x224': {'target_acc': 85.0},
           'hiera_large_16x224': {'target_acc': 87.3},
           'hiera_huge_16x224': {'target_acc': 87.8}}}

K400_CLASSES = ['abseiling',
 'air drumming',
 'answering questions',
 'applauding',
 'applying cream',
 'archery',
 'arm wrestling',
 'arranging flowers',
 'assembling computer',
 'auctioning',
 'baby waking up',
 'baking cookies',
 'balloon blowing',
 'bandaging',
 'barbequing',
 'bartending',
 'beatboxing',
 'bee keeping',
 'belly dancing',
 'bench pressing',
 'bending back',
 'bending metal',
 'biking through snow',
 'blasting sand',
 'blowing glass',
 'blowing leaves',
 'blowing nose',
 'blowing out candles',
 'bobsledding',
 'bookbinding',
 'bouncing on trampoline',
 'bowling',
 'braiding hair',
 'breading or breadcrumbing',
 'breakdancing',
 'brush painting',
 'brushing hair',
 'brushing teeth',
 'building cabinet',
 'building shed',
 'bungee jumping',
 'busking',
 'canoeing or kayaking',
 'capoeira',
 'carrying baby',
 'cartwheeling',
 'carving pumpkin',
 'catching fish',
 'catching or throwing baseball',
 'catching or throwing frisbee',
 'catching or throwing softball',
 'celebrating',
 'changing oil',
 'changing wheel',
 'checking tires',
 'cheerleading',
 'chopping wood',
 'clapping',
 'clay pottery making',
 'clean and jerk',
 'cleaning floor',
 'cleaning gutters',
 'cleaning pool',
 'cleaning shoes',
 'cleaning toilet',
 'cleaning windows',
 'climbing a rope',
 'climbing ladder',
 'climbing tree',
 'contact juggling',
 'cooking chicken',
 'cooking egg',
 'cooking on campfire',
 'cooking sausages',
 'counting money',
 'country line dancing',
 'cracking neck',
 'crawling baby',
 'crossing river',
 'crying',
 'curling hair',
 'cutting nails',
 'cutting pineapple',
 'cutting watermelon',
 'dancing ballet',
 'dancing charleston',
 'dancing gangnam style',
 'dancing macarena',
 'deadlifting',
 'decorating the christmas tree',
 'digging',
 'dining',
 'disc golfing',
 'diving cliff',
 'dodgeball',
 'doing aerobics',
 'doing laundry',
 'doing nails',
 'drawing',
 'dribbling basketball',
 'drinking',
 'drinking beer',
 'drinking shots',
 'driving car',
 'driving tractor',
 'drop kicking',
 'drumming fingers',
 'dunking basketball',
 'dying hair',
 'eating burger',
 'eating cake',
 'eating carrots',
 'eating chips',
 'eating doughnuts',
 'eating hotdog',
 'eating ice cream',
 'eating spaghetti',
 'eating watermelon',
 'egg hunting',
 'exercising arm',
 'exercising with an exercise ball',
 'extinguishing fire',
 'faceplanting',
 'feeding birds',
 'feeding fish',
 'feeding goats',
 'filling eyebrows',
 'finger snapping',
 'fixing hair',
 'flipping pancake',
 'flying kite',
 'folding clothes',
 'folding napkins',
 'folding paper',
 'front raises',
 'frying vegetables',
 'garbage collecting',
 'gargling',
 'getting a haircut',
 'getting a tattoo',
 'giving or receiving award',
 'golf chipping',
 'golf driving',
 'golf putting',
 'grinding meat',
 'grooming dog',
 'grooming horse',
 'gymnastics tumbling',
 'hammer throw',
 'headbanging',
 'headbutting',
 'high jump',
 'high kick',
 'hitting baseball',
 'hockey stop',
 'holding snake',
 'hopscotch',
 'hoverboarding',
 'hugging',
 'hula hooping',
 'hurdling',
 'hurling (sport)',
 'ice climbing',
 'ice fishing',
 'ice skating',
 'ironing',
 'javelin throw',
 'jetskiing',
 'jogging',
 'juggling balls',
 'juggling fire',
 'juggling soccer ball',
 'jumping into pool',
 'jumpstyle dancing',
 'kicking field goal',
 'kicking soccer ball',
 'kissing',
 'kitesurfing',
 'knitting',
 'krumping',
 'laughing',
 'laying bricks',
 'long jump',
 'lunge',
 'making a cake',
 'making a sandwich',
 'making bed',
 'making jewelry',
 'making pizza',
 'making snowman',
 'making sushi',
 'making tea',
 'marching',
 'massaging back',
 'massaging feet',
 'massaging legs',
 "massaging person's head",
 'milking cow',
 'mopping floor',
 'motorcycling',
 'moving furniture',
 'mowing lawn',
 'news anchoring',
 'opening bottle',
 'opening present',
 'paragliding',
 'parasailing',
 'parkour',
 'passing American football (in game)',
 'passing American football (not in game)',
 'peeling apples',
 'peeling potatoes',
 'petting animal (not cat)',
 'petting cat',
 'picking fruit',
 'planting trees',
 'plastering',
 'playing accordion',
 'playing badminton',
 'playing bagpipes',
 'playing basketball',
 'playing bass guitar',
 'playing cards',
 'playing cello',
 'playing chess',
 'playing clarinet',
 'playing controller',
 'playing cricket',
 'playing cymbals',
 'playing didgeridoo',
 'playing drums',
 'playing flute',
 'playing guitar',
 'playing harmonica',
 'playing harp',
 'playing ice hockey',
 'playing keyboard',
 'playing kickball',
 'playing monopoly',
 'playing organ',
 'playing paintball',
 'playing piano',
 'playing poker',
 'playing recorder',
 'playing saxophone',
 'playing squash or racquetball',
 'playing tennis',
 'playing trombone',
 'playing trumpet',
 'playing ukulele',
 'playing violin',
 'playing volleyball',
 'playing xylophone',
 'pole vault',
 'presenting weather forecast',
 'pull ups',
 'pumping fist',
 'pumping gas',
 'punching bag',
 'punching person (boxing)',
 'push up',
 'pushing car',
 'pushing cart',
 'pushing wheelchair',
 'reading book',
 'reading newspaper',
 'recording music',
 'riding a bike',
 'riding camel',
 'riding elephant',
 'riding mechanical bull',
 'riding mountain bike',
 'riding mule',
 'riding or walking with horse',
 'riding scooter',
 'riding unicycle',
 'ripping paper',
 'robot dancing',
 'rock climbing',
 'rock scissors paper',
 'roller skating',
 'running on treadmill',
 'sailing',
 'salsa dancing',
 'sanding floor',
 'scrambling eggs',
 'scuba diving',
 'setting table',
 'shaking hands',
 'shaking head',
 'sharpening knives',
 'sharpening pencil',
 'shaving head',
 'shaving legs',
 'shearing sheep',
 'shining shoes',
 'shooting basketball',
 'shooting goal (soccer)',
 'shot put',
 'shoveling snow',
 'shredding paper',
 'shuffling cards',
 'side kick',
 'sign language interpreting',
 'singing',
 'situp',
 'skateboarding',
 'ski jumping',
 'skiing (not slalom or crosscountry)',
 'skiing crosscountry',
 'skiing slalom',
 'skipping rope',
 'skydiving',
 'slacklining',
 'slapping',
 'sled dog racing',
 'smoking',
 'smoking hookah',
 'snatch weight lifting',
 'sneezing',
 'sniffing',
 'snorkeling',
 'snowboarding',
 'snowkiting',
 'snowmobiling',
 'somersaulting',
 'spinning poi',
 'spray painting',
 'spraying',
 'springboard diving',
 'squat',
 'sticking tongue out',
 'stomping grapes',
 'stretching arm',
 'stretching leg',
 'strumming guitar',
 'surfing crowd',
 'surfing water',
 'sweeping floor',
 'swimming backstroke',
 'swimming breast stroke',
 'swimming butterfly stroke',
 'swing dancing',
 'swinging legs',
 'swinging on something',
 'sword fighting',
 'tai chi',
 'taking a shower',
 'tango dancing',
 'tap dancing',
 'tapping guitar',
 'tapping pen',
 'tasting beer',
 'tasting food',
 'testifying',
 'texting',
 'throwing axe',
 'throwing ball',
 'throwing discus',
 'tickling',
 'tobogganing',
 'tossing coin',
 'tossing salad',
 'training dog',
 'trapezing',
 'trimming or shaving beard',
 'trimming trees',
 'triple jump',
 'tying bow tie',
 'tying knot (not on a tie)',
 'tying tie',
 'unboxing',
 'unloading truck',
 'using computer',
 'using remote controller (not gaming)',
 'using segway',
 'vault',
 'waiting in line',
 'walking the dog',
 'washing dishes',
 'washing feet',
 'washing hair',
 'washing hands',
 'water skiing',
 'water sliding',
 'watering plants',
 'waxing back',
 'waxing chest',
 'waxing eyebrows',
 'waxing legs',
 'weaving basket',
 'welding',
 'whistling',
 'windsurfing',
 'wrapping present',
 'wrestling',
 'writing',
 'yawning',
 'yoga',
 'zumba']

SYNSET_TO_IDX = {'n01440764': 0,
 'n01443537': 1,
 'n01484850': 2,
 'n01491361': 3,
 'n01494475': 4,
 'n01496331': 5,
 'n01498041': 6,
 'n01514668': 7,
 'n01514859': 8,
 'n01518878': 9,
 'n01530575': 10,
 'n01531178': 11,
 'n01532829': 12,
 'n01534433': 13,
 'n01537544': 14,
 'n01558993': 15,
 'n01560419': 16,
 'n01580077': 17,
 'n01582220': 18,
 'n01592084': 19,
 'n01601694': 20,
 'n01608432': 21,
 'n01614925': 22,
 'n01616318': 23,
 'n01622779': 24,
 'n01629819': 25,
 'n01630670': 26,
 'n01631663': 27,
 'n01632458': 28,
 'n01632777': 29,
 'n01641577': 30,
 'n01644373': 31,
 'n01644900': 32,
 'n01664065': 33,
 'n01665541': 34,
 'n01667114': 35,
 'n01667778': 36,
 'n01669191': 37,
 'n01675722': 38,
 'n01677366': 39,
 'n01682714': 40,
 'n01685808': 41,
 'n01687978': 42,
 'n01688243': 43,
 'n01689811': 44,
 'n01692333': 45,
 'n01693334': 46,
 'n01694178': 47,
 'n01695060': 48,
 'n01697457': 49,
 'n01698640': 50,
 'n01704323': 51,
 'n01728572': 52,
 'n01728920': 53,
 'n01729322': 54,
 'n01729977': 55,
 'n01734418': 56,
 'n01735189': 57,
 'n01737021': 58,
 'n01739381': 59,
 'n01740131': 60,
 'n01742172': 61,
 'n01744401': 62,
 'n01748264': 63,
 'n01749939': 64,
 'n01751748': 65,
 'n01753488': 66,
 'n01755581': 67,
 'n01756291': 68,
 'n01768244': 69,
 'n01770081': 70,
 'n01770393': 71,
 'n01773157': 72,
 'n01773549': 73,
 'n01773797': 74,
 'n01774384': 75,
 'n01774750': 76,
 'n01775062': 77,
 'n01776313': 78,
 'n01784675': 79,
 'n01795545': 80,
 'n01796340': 81,
 'n01797886': 82,
 'n01798484': 83,
 'n01806143': 84,
 'n01806567': 85,
 'n01807496': 86,
 'n01817953': 87,
 'n01818515': 88,
 'n01819313': 89,
 'n01820546': 90,
 'n01824575': 91,
 'n01828970': 92,
 'n01829413': 93,
 'n01833805': 94,
 'n01843065': 95,
 'n01843383': 96,
 'n01847000': 97,
 'n01855032': 98,
 'n01855672': 99,
 'n01860187': 100,
 'n01871265': 101,
 'n01872401': 102,
 'n01873310': 103,
 'n01877812': 104,
 'n01882714': 105,
 'n01883070': 106,
 'n01910747': 107,
 'n01914609': 108,
 'n01917289': 109,
 'n01924916': 110,
 'n01930112': 111,
 'n01943899': 112,
 'n01944390': 113,
 'n01945685': 114,
 'n01950731': 115,
 'n01955084': 116,
 'n01968897': 117,
 'n01978287': 118,
 'n01978455': 119,
 'n01980166': 120,
 'n01981276': 121,
 'n01983481': 122,
 'n01984695': 123,
 'n01985128': 124,
 'n01986214': 125,
 'n01990800': 126,
 'n02002556': 127,
 'n02002724': 128,
 'n02006656': 129,
 'n02007558': 130,
 'n02009229': 131,
 'n02009912': 132,
 'n02011460': 133,
 'n02012849': 134,
 'n02013706': 135,
 'n02017213': 136,
 'n02018207': 137,
 'n02018795': 138,
 'n02025239': 139,
 'n02027492': 140,
 'n02028035': 141,
 'n02033041': 142,
 'n02037110': 143,
 'n02051845': 144,
 'n02056570': 145,
 'n02058221': 146,
 'n02066245': 147,
 'n02071294': 148,
 'n02074367': 149,
 'n02077923': 150,
 'n02085620': 151,
 'n02085782': 152,
 'n02085936': 153,
 'n02086079': 154,
 'n02086240': 155,
 'n02086646': 156,
 'n02086910': 157,
 'n02087046': 158,
 'n02087394': 159,
 'n02088094': 160,
 'n02088238': 161,
 'n02088364': 162,
 'n02088466': 163,
 'n02088632': 164,
 'n02089078': 165,
 'n02089867': 166,
 'n02089973': 167,
 'n02090379': 168,
 'n02090622': 169,
 'n02090721': 170,
 'n02091032': 171,
 'n02091134': 172,
 'n02091244': 173,
 'n02091467': 174,
 'n02091635': 175,
 'n02091831': 176,
 'n02092002': 177,
 'n02092339': 178,
 'n02093256': 179,
 'n02093428': 180,
 'n02093647': 181,
 'n02093754': 182,
 'n02093859': 183,
 'n02093991': 184,
 'n02094114': 185,
 'n02094258': 186,
 'n02094433': 187,
 'n02095314': 188,
 'n02095570': 189,
 'n02095889': 190,
 'n02096051': 191,
 'n02096177': 192,
 'n02096294': 193,
 'n02096437': 194,
 'n02096585': 195,
 'n02097047': 196,
 'n02097130': 197,
 'n02097209': 198,
 'n02097298': 199,
 'n02097474': 200,
 'n02097658': 201,
 'n02098105': 202,
 'n02098286': 203,
 'n02098413': 204,
 'n02099267': 205,
 'n02099429': 206,
 'n02099601': 207,
 'n02099712': 208,
 'n02099849': 209,
 'n02100236': 210,
 'n02100583': 211,
 'n02100735': 212,
 'n02100877': 213,
 'n02101006': 214,
 'n02101388': 215,
 'n02101556': 216,
 'n02102040': 217,
 'n02102177': 218,
 'n02102318': 219,
 'n02102480': 220,
 'n02102973': 221,
 'n02104029': 222,
 'n02104365': 223,
 'n02105056': 224,
 'n02105162': 225,
 'n02105251': 226,
 'n02105412': 227,
 'n02105505': 228,
 'n02105641': 229,
 'n02105855': 230,
 'n02106030': 231,
 'n02106166': 232,
 'n02106382': 233,
 'n02106550': 234,
 'n02106662': 235,
 'n02107142': 236,
 'n02107312': 237,
 'n02107574': 238,
 'n02107683': 239,
 'n02107908': 240,
 'n02108000': 241,
 'n02108089': 242,
 'n02108422': 243,
 'n02108551': 244,
 'n02108915': 245,
 'n02109047': 246,
 'n02109525': 247,
 'n02109961': 248,
 'n02110063': 249,
 'n02110185': 250,
 'n02110341': 251,
 'n02110627': 252,
 'n02110806': 253,
 'n02110958': 254,
 'n02111129': 255,
 'n02111277': 256,
 'n02111500': 257,
 'n02111889': 258,
 'n02112018': 259,
 'n02112137': 260,
 'n02112350': 261,
 'n02112706': 262,
 'n02113023': 263,
 'n02113186': 264,
 'n02113624': 265,
 'n02113712': 266,
 'n02113799': 267,
 'n02113978': 268,
 'n02114367': 269,
 'n02114548': 270,
 'n02114712': 271,
 'n02114855': 272,
 'n02115641': 273,
 'n02115913': 274,
 'n02116738': 275,
 'n02117135': 276,
 'n02119022': 277,
 'n02119789': 278,
 'n02120079': 279,
 'n02120505': 280,
 'n02123045': 281,
 'n02123159': 282,
 'n02123394': 283,
 'n02123597': 284,
 'n02124075': 285,
 'n02125311': 286,
 'n02127052': 287,
 'n02128385': 288,
 'n02128757': 289,
 'n02128925': 290,
 'n02129165': 291,
 'n02129604': 292,
 'n02130308': 293,
 'n02132136': 294,
 'n02133161': 295,
 'n02134084': 296,
 'n02134418': 297,
 'n02137549': 298,
 'n02138441': 299,
 'n02165105': 300,
 'n02165456': 301,
 'n02167151': 302,
 'n02168699': 303,
 'n02169497': 304,
 'n02172182': 305,
 'n02174001': 306,
 'n02177972': 307,
 'n02190166': 308,
 'n02206856': 309,
 'n02219486': 310,
 'n02226429': 311,
 'n02229544': 312,
 'n02231487': 313,
 'n02233338': 314,
 'n02236044': 315,
 'n02256656': 316,
 'n02259212': 317,
 'n02264363': 318,
 'n02268443': 319,
 'n02268853': 320,
 'n02276258': 321,
 'n02277742': 322,
 'n02279972': 323,
 'n02280649': 324,
 'n02281406': 325,
 'n02281787': 326,
 'n02317335': 327,
 'n02319095': 328,
 'n02321529': 329,
 'n02325366': 330,
 'n02326432': 331,
 'n02328150': 332,
 'n02342885': 333,
 'n02346627': 334,
 'n02356798': 335,
 'n02361337': 336,
 'n02363005': 337,
 'n02364673': 338,
 'n02389026': 339,
 'n02391049': 340,
 'n02395406': 341,
 'n02396427': 342,
 'n02397096': 343,
 'n02398521': 344,
 'n02403003': 345,
 'n02408429': 346,
 'n02410509': 347,
 'n02412080': 348,
 'n02415577': 349,
 'n02417914': 350,
 'n02422106': 351,
 'n02422699': 352,
 'n02423022': 353,
 'n02437312': 354,
 'n02437616': 355,
 'n02441942': 356,
 'n02442845': 357,
 'n02443114': 358,
 'n02443484': 359,
 'n02444819': 360,
 'n02445715': 361,
 'n02447366': 362,
 'n02454379': 363,
 'n02457408': 364,
 'n02480495': 365,
 'n02480855': 366,
 'n02481823': 367,
 'n02483362': 368,
 'n02483708': 369,
 'n02484975': 370,
 'n02486261': 371,
 'n02486410': 372,
 'n02487347': 373,
 'n02488291': 374,
 'n02488702': 375,
 'n02489166': 376,
 'n02490219': 377,
 'n02492035': 378,
 'n02492660': 379,
 'n02493509': 380,
 'n02493793': 381,
 'n02494079': 382,
 'n02497673': 383,
 'n02500267': 384,
 'n02504013': 385,
 'n02504458': 386,
 'n02509815': 387,
 'n02510455': 388,
 'n02514041': 389,
 'n02526121': 390,
 'n02536864': 391,
 'n02606052': 392,
 'n02607072': 393,
 'n02640242': 394,
 'n02641379': 395,
 'n02643566': 396,
 'n02655020': 397,
 'n02666196': 398,
 'n02667093': 399,
 'n02669723': 400,
 'n02672831': 401,
 'n02676566': 402,
 'n02687172': 403,
 'n02690373': 404,
 'n02692877': 405,
 'n02699494': 406,
 'n02701002': 407,
 'n02704792': 408,
 'n02708093': 409,
 'n02727426': 410,
 'n02730930': 411,
 'n02747177': 412,
 'n02749479': 413,
 'n02769748': 414,
 'n02776631': 415,
 'n02777292': 416,
 'n02782093': 417,
 'n02783161': 418,
 'n02786058': 419,
 'n02787622': 420,
 'n02788148': 421,
 'n02790996': 422,
 'n02791124': 423,
 'n02791270': 424,
 'n02793495': 425,
 'n02794156': 426,
 'n02795169': 427,
 'n02797295': 428,
 'n02799071': 429,
 'n02802426': 430,
 'n02804414': 431,
 'n02804610': 432,
 'n02807133': 433,
 'n02808304': 434,
 'n02808440': 435,
 'n02814533': 436,
 'n02814860': 437,
 'n02815834': 438,
 'n02817516': 439,
 'n02823428': 440,
 'n02823750': 441,
 'n02825657': 442,
 'n02834397': 443,
 'n02835271': 444,
 'n02837789': 445,
 'n02840245': 446,
 'n02841315': 447,
 'n02843684': 448,
 'n02859443': 449,
 'n02860847': 450,
 'n02865351': 451,
 'n02869837': 452,
 'n02870880': 453,
 'n02871525': 454,
 'n02877765': 455,
 'n02879718': 456,
 'n02883205': 457,
 'n02892201': 458,
 'n02892767': 459,
 'n02894605': 460,
 'n02895154': 461,
 'n02906734': 462,
 'n02909870': 463,
 'n02910353': 464,
 'n02916936': 465,
 'n02917067': 466,
 'n02927161': 467,
 'n02930766': 468,
 'n02939185': 469,
 'n02948072': 470,
 'n02950826': 471,
 'n02951358': 472,
 'n02951585': 473,
 'n02963159': 474,
 'n02965783': 475,
 'n02966193': 476,
 'n02966687': 477,
 'n02971356': 478,
 'n02974003': 479,
 'n02977058': 480,
 'n02978881': 481,
 'n02979186': 482,
 'n02980441': 483,
 'n02981792': 484,
 'n02988304': 485,
 'n02992211': 486,
 'n02992529': 487,
 'n02999410': 488,
 'n03000134': 489,
 'n03000247': 490,
 'n03000684': 491,
 'n03014705': 492,
 'n03016953': 493,
 'n03017168': 494,
 'n03018349': 495,
 'n03026506': 496,
 'n03028079': 497,
 'n03032252': 498,
 'n03041632': 499,
 'n03042490': 500,
 'n03045698': 501,
 'n03047690': 502,
 'n03062245': 503,
 'n03063599': 504,
 'n03063689': 505,
 'n03065424': 506,
 'n03075370': 507,
 'n03085013': 508,
 'n03089624': 509,
 'n03095699': 510,
 'n03100240': 511,
 'n03109150': 512,
 'n03110669': 513,
 'n03124043': 514,
 'n03124170': 515,
 'n03125729': 516,
 'n03126707': 517,
 'n03127747': 518,
 'n03127925': 519,
 'n03131574': 520,
 'n03133878': 521,
 'n03134739': 522,
 'n03141823': 523,
 'n03146219': 524,
 'n03160309': 525,
 'n03179701': 526,
 'n03180011': 527,
 'n03187595': 528,
 'n03188531': 529,
 'n03196217': 530,
 'n03197337': 531,
 'n03201208': 532,
 'n03207743': 533,
 'n03207941': 534,
 'n03208938': 535,
 'n03216828': 536,
 'n03218198': 537,
 'n03220513': 538,
 'n03223299': 539,
 'n03240683': 540,
 'n03249569': 541,
 'n03250847': 542,
 'n03255030': 543,
 'n03259280': 544,
 'n03271574': 545,
 'n03272010': 546,
 'n03272562': 547,
 'n03290653': 548,
 'n03291819': 549,
 'n03297495': 550,
 'n03314780': 551,
 'n03325584': 552,
 'n03337140': 553,
 'n03344393': 554,
 'n03345487': 555,
 'n03347037': 556,
 'n03355925': 557,
 'n03372029': 558,
 'n03376595': 559,
 'n03379051': 560,
 'n03384352': 561,
 'n03388043': 562,
 'n03388183': 563,
 'n03388549': 564,
 'n03393912': 565,
 'n03394916': 566,
 'n03400231': 567,
 'n03404251': 568,
 'n03417042': 569,
 'n03424325': 570,
 'n03425413': 571,
 'n03443371': 572,
 'n03444034': 573,
 'n03445777': 574,
 'n03445924': 575,
 'n03447447': 576,
 'n03447721': 577,
 'n03450230': 578,
 'n03452741': 579,
 'n03457902': 580,
 'n03459775': 581,
 'n03461385': 582,
 'n03467068': 583,
 'n03476684': 584,
 'n03476991': 585,
 'n03478589': 586,
 'n03481172': 587,
 'n03482405': 588,
 'n03483316': 589,
 'n03485407': 590,
 'n03485794': 591,
 'n03492542': 592,
 'n03494278': 593,
 'n03495258': 594,
 'n03496892': 595,
 'n03498962': 596,
 'n03527444': 597,
 'n03529860': 598,
 'n03530642': 599,
 'n03532672': 600,
 'n03534580': 601,
 'n03535780': 602,
 'n03538406': 603,
 'n03544143': 604,
 'n03584254': 605,
 'n03584829': 606,
 'n03590841': 607,
 'n03594734': 608,
 'n03594945': 609,
 'n03595614': 610,
 'n03598930': 611,
 'n03599486': 612,
 'n03602883': 613,
 'n03617480': 614,
 'n03623198': 615,
 'n03627232': 616,
 'n03630383': 617,
 'n03633091': 618,
 'n03637318': 619,
 'n03642806': 620,
 'n03649909': 621,
 'n03657121': 622,
 'n03658185': 623,
 'n03661043': 624,
 'n03662601': 625,
 'n03666591': 626,
 'n03670208': 627,
 'n03673027': 628,
 'n03676483': 629,
 'n03680355': 630,
 'n03690938': 631,
 'n03691459': 632,
 'n03692522': 633,
 'n03697007': 634,
 'n03706229': 635,
 'n03709823': 636,
 'n03710193': 637,
 'n03710637': 638,
 'n03710721': 639,
 'n03717622': 640,
 'n03720891': 641,
 'n03721384': 642,
 'n03724870': 643,
 'n03729826': 644,
 'n03733131': 645,
 'n03733281': 646,
 'n03733805': 647,
 'n03742115': 648,
 'n03743016': 649,
 'n03759954': 650,
 'n03761084': 651,
 'n03763968': 652,
 'n03764736': 653,
 'n03769881': 654,
 'n03770439': 655,
 'n03770679': 656,
 'n03773504': 657,
 'n03775071': 658,
 'n03775546': 659,
 'n03776460': 660,
 'n03777568': 661,
 'n03777754': 662,
 'n03781244': 663,
 'n03782006': 664,
 'n03785016': 665,
 'n03786901': 666,
 'n03787032': 667,
 'n03788195': 668,
 'n03788365': 669,
 'n03791053': 670,
 'n03792782': 671,
 'n03792972': 672,
 'n03793489': 673,
 'n03794056': 674,
 'n03796401': 675,
 'n03803284': 676,
 'n03804744': 677,
 'n03814639': 678,
 'n03814906': 679,
 'n03825788': 680,
 'n03832673': 681,
 'n03837869': 682,
 'n03838899': 683,
 'n03840681': 684,
 'n03841143': 685,
 'n03843555': 686,
 'n03854065': 687,
 'n03857828': 688,
 'n03866082': 689,
 'n03868242': 690,
 'n03868863': 691,
 'n03871628': 692,
 'n03873416': 693,
 'n03874293': 694,
 'n03874599': 695,
 'n03876231': 696,
 'n03877472': 697,
 'n03877845': 698,
 'n03884397': 699,
 'n03887697': 700,
 'n03888257': 701,
 'n03888605': 702,
 'n03891251': 703,
 'n03891332': 704,
 'n03895866': 705,
 'n03899768': 706,
 'n03902125': 707,
 'n03903868': 708,
 'n03908618': 709,
 'n03908714': 710,
 'n03916031': 711,
 'n03920288': 712,
 'n03924679': 713,
 'n03929660': 714,
 'n03929855': 715,
 'n03930313': 716,
 'n03930630': 717,
 'n03933933': 718,
 'n03935335': 719,
 'n03937543': 720,
 'n03938244': 721,
 'n03942813': 722,
 'n03944341': 723,
 'n03947888': 724,
 'n03950228': 725,
 'n03954731': 726,
 'n03956157': 727,
 'n03958227': 728,
 'n03961711': 729,
 'n03967562': 730,
 'n03970156': 731,
 'n03976467': 732,
 'n03976657': 733,
 'n03977966': 734,
 'n03980874': 735,
 'n03982430': 736,
 'n03983396': 737,
 'n03991062': 738,
 'n03992509': 739,
 'n03995372': 740,
 'n03998194': 741,
 'n04004767': 742,
 'n04005630': 743,
 'n04008634': 744,
 'n04009552': 745,
 'n04019541': 746,
 'n04023962': 747,
 'n04026417': 748,
 'n04033901': 749,
 'n04033995': 750,
 'n04037443': 751,
 'n04039381': 752,
 'n04040759': 753,
 'n04041544': 754,
 'n04044716': 755,
 'n04049303': 756,
 'n04065272': 757,
 'n04067472': 758,
 'n04069434': 759,
 'n04070727': 760,
 'n04074963': 761,
 'n04081281': 762,
 'n04086273': 763,
 'n04090263': 764,
 'n04099969': 765,
 'n04111531': 766,
 'n04116512': 767,
 'n04118538': 768,
 'n04118776': 769,
 'n04120489': 770,
 'n04125021': 771,
 'n04127249': 772,
 'n04131690': 773,
 'n04133789': 774,
 'n04136333': 775,
 'n04141076': 776,
 'n04141327': 777,
 'n04141975': 778,
 'n04146614': 779,
 'n04147183': 780,
 'n04149813': 781,
 'n04152593': 782,
 'n04153751': 783,
 'n04154565': 784,
 'n04162706': 785,
 'n04179913': 786,
 'n04192698': 787,
 'n04200800': 788,
 'n04201297': 789,
 'n04204238': 790,
 'n04204347': 791,
 'n04208210': 792,
 'n04209133': 793,
 'n04209239': 794,
 'n04228054': 795,
 'n04229816': 796,
 'n04235860': 797,
 'n04238763': 798,
 'n04239074': 799,
 'n04243546': 800,
 'n04251144': 801,
 'n04252077': 802,
 'n04252225': 803,
 'n04254120': 804,
 'n04254680': 805,
 'n04254777': 806,
 'n04258138': 807,
 'n04259630': 808,
 'n04263257': 809,
 'n04264628': 810,
 'n04265275': 811,
 'n04266014': 812,
 'n04270147': 813,
 'n04273569': 814,
 'n04275548': 815,
 'n04277352': 816,
 'n04285008': 817,
 'n04286575': 818,
 'n04296562': 819,
 'n04310018': 820,
 'n04311004': 821,
 'n04311174': 822,
 'n04317175': 823,
 'n04325704': 824,
 'n04326547': 825,
 'n04328186': 826,
 'n04330267': 827,
 'n04332243': 828,
 'n04335435': 829,
 'n04336792': 830,
 'n04344873': 831,
 'n04346328': 832,
 'n04347754': 833,
 'n04350905': 834,
 'n04355338': 835,
 'n04355933': 836,
 'n04356056': 837,
 'n04357314': 838,
 'n04366367': 839,
 'n04367480': 840,
 'n04370456': 841,
 'n04371430': 842,
 'n04371774': 843,
 'n04372370': 844,
 'n04376876': 845,
 'n04380533': 846,
 'n04389033': 847,
 'n04392985': 848,
 'n04398044': 849,
 'n04399382': 850,
 'n04404412': 851,
 'n04409515': 852,
 'n04417672': 853,
 'n04418357': 854,
 'n04423845': 855,
 'n04428191': 856,
 'n04429376': 857,
 'n04435653': 858,
 'n04442312': 859,
 'n04443257': 860,
 'n04447861': 861,
 'n04456115': 862,
 'n04458633': 863,
 'n04461696': 864,
 'n04462240': 865,
 'n04465501': 866,
 'n04467665': 867,
 'n04476259': 868,
 'n04479046': 869,
 'n04482393': 870,
 'n04483307': 871,
 'n04485082': 872,
 'n04486054': 873,
 'n04487081': 874,
 'n04487394': 875,
 'n04493381': 876,
 'n04501370': 877,
 'n04505470': 878,
 'n04507155': 879,
 'n04509417': 880,
 'n04515003': 881,
 'n04517823': 882,
 'n04522168': 883,
 'n04523525': 884,
 'n04525038': 885,
 'n04525305': 886,
 'n04532106': 887,
 'n04532670': 888,
 'n04536866': 889,
 'n04540053': 890,
 'n04542943': 891,
 'n04548280': 892,
 'n04548362': 893,
 'n04550184': 894,
 'n04552348': 895,
 'n04553703': 896,
 'n04554684': 897,
 'n04557648': 898,
 'n04560804': 899,
 'n04562935': 900,
 'n04579145': 901,
 'n04579432': 902,
 'n04584207': 903,
 'n04589890': 904,
 'n04590129': 905,
 'n04591157': 906,
 'n04591713': 907,
 'n04592741': 908,
 'n04596742': 909,
 'n04597913': 910,
 'n04599235': 911,
 'n04604644': 912,
 'n04606251': 913,
 'n04612504': 914,
 'n04613696': 915,
 'n06359193': 916,
 'n06596364': 917,
 'n06785654': 918,
 'n06794110': 919,
 'n06874185': 920,
 'n07248320': 921,
 'n07565083': 922,
 'n07579787': 923,
 'n07583066': 924,
 'n07584110': 925,
 'n07590611': 926,
 'n07613480': 927,
 'n07614500': 928,
 'n07615774': 929,
 'n07684084': 930,
 'n07693725': 931,
 'n07695742': 932,
 'n07697313': 933,
 'n07697537': 934,
 'n07711569': 935,
 'n07714571': 936,
 'n07714990': 937,
 'n07715103': 938,
 'n07716358': 939,
 'n07716906': 940,
 'n07717410': 941,
 'n07717556': 942,
 'n07718472': 943,
 'n07718747': 944,
 'n07720875': 945,
 'n07730033': 946,
 'n07734744': 947,
 'n07742313': 948,
 'n07745940': 949,
 'n07747607': 950,
 'n07749582': 951,
 'n07753113': 952,
 'n07753275': 953,
 'n07753592': 954,
 'n07754684': 955,
 'n07760859': 956,
 'n07768694': 957,
 'n07802026': 958,
 'n07831146': 959,
 'n07836838': 960,
 'n07860988': 961,
 'n07871810': 962,
 'n07873807': 963,
 'n07875152': 964,
 'n07880968': 965,
 'n07892512': 966,
 'n07920052': 967,
 'n07930864': 968,
 'n07932039': 969,
 'n09193705': 970,
 'n09229709': 971,
 'n09246464': 972,
 'n09256479': 973,
 'n09288635': 974,
 'n09332890': 975,
 'n09399592': 976,
 'n09421951': 977,
 'n09428293': 978,
 'n09468604': 979,
 'n09472597': 980,
 'n09835506': 981,
 'n10148035': 982,
 'n10565667': 983,
 'n11879895': 984,
 'n11939491': 985,
 'n12057211': 986,
 'n12144580': 987,
 'n12267677': 988,
 'n12620546': 989,
 'n12768682': 990,
 'n12985857': 991,
 'n12998815': 992,
 'n13037406': 993,
 'n13040303': 994,
 'n13044778': 995,
 'n13052670': 996,
 'n13054560': 997,
 'n13133613': 998,
 'n15075141': 999}

IMAGE_MODEL_CONFIGS = MODEL_CONFIGS.get("image", {})
VIDEO_MODEL_CONFIGS = MODEL_CONFIGS.get("video", {})
def get_model(model_name):
    print(f"Loading {model_name}...")
    
    model_func = MODEL_BUILDERS.get(model_name)
    if model_func is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    if "16x224" in model_name:
        model = model_func(pretrained=True, checkpoint="mae_k400_ft_k400")
    else:
        model = model_func(pretrained=True, checkpoint="mae_in1k_ft_in1k")
    
    return model

def get_synset_mapping(data_dir):
    return SYNSET_TO_IDX

def evaluate_image_model(model_name, data_dir, batch_size, device, num_images):
    print(f"Device: {device}")
    if device != "cpu":
        paddle.device.set_device(device)
    
    model = get_model(model_name)
    model.eval()
    target_acc = IMAGE_MODEL_CONFIGS[model_name]["target_acc"]
    
    synset_to_idx = get_synset_mapping(data_dir)
    if not synset_to_idx:
        print(f"No synset mapping found")
        return None
    
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize(256),
        paddle.vision.transforms.CenterCrop(224),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    class_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("n")
    ])
    
    image_files = []
    for class_dir in class_dirs:
        if class_dir not in synset_to_idx:
            continue
        gt_idx = synset_to_idx[class_dir]
        class_path = os.path.join(data_dir, class_dir)
        files = [
            (os.path.join(class_path, f), gt_idx)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_files.extend(files)
    
    if num_images > 0:
        image_files = image_files[:num_images]
        print(f"Limiting to {num_images} images for testing...")
    
    if not image_files:
        print(f"No images found in {data_dir}")
        return None
    
    print(f"Total images: {len(image_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    with paddle.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size), desc=f"Evaluating {model_name}"):
            batch_files = image_files[i:i+batch_size]
            images = []
            labels_batch = []
            
            for img_path, gt_idx in batch_files:
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_tensor = transform(img)
                    images.append(img_tensor)
                    labels_batch.append(gt_idx)
                except Exception as e:
                    continue
            
            if not images:
                continue
            
            images = paddle.stack(images)
            outputs = model(images)
            
            _, top1_pred = outputs._max(1)
            top1_pred = top1_pred.numpy()
            top5 = outputs.topk(5, axis=1)
            top5_pred = top5.indices.numpy()
            
            total += len(labels_batch)
            for idx, label in enumerate(labels_batch):
                if top1_pred[idx] == label:
                    correct_top1 += 1
                if label in top5_pred[idx]:
                    correct_top5 += 1
    
    if total == 0:
        return None
    
    return {
        "model": model_name,
        "type": "image",
        "top1": 100.0 * correct_top1 / total,
        "top5": 100.0 * correct_top5 / total,
        "total": total,
        "target": target_acc
    }

def load_video_frames(video_path, num_frames=16):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = list(range(total_frames))
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(Image.fromarray(frame))
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.new('RGB', (224, 224), (0, 0, 0)))
    
    cap.release()
    return frames

def transform_video_frames(frames):
    transform = paddle.vision.transforms.Compose([
        paddle.vision.transforms.Resize(256),
        paddle.vision.transforms.CenterCrop(224),
        paddle.vision.transforms.ToTensor(),
        paddle.vision.transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])
    
    tensors = [transform(f) for f in frames]
    video_tensor = paddle.stack(tensors, axis=1)
    return video_tensor

def evaluate_video_model(model_name, data_dir, batch_size, device, num_videos):
    print(f"Device: {device}")
    if device != "cpu":
        paddle.device.set_device(device)
    
    model = get_model(model_name)
    model.eval()
    target_acc = VIDEO_MODEL_CONFIGS[model_name]["target_acc"]
    
    video_files = []
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            for f in os.listdir(class_path):
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append((os.path.join(class_path, f), class_name))
    
    if num_videos > 0:
        video_files = video_files[:num_videos]
        print(f"Limiting to {num_videos} videos for testing...")
    
    if not video_files:
        print(f"No videos found in {data_dir}")
        return None
    
    print(f"Total videos: {len(video_files)}")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    for i in tqdm(range(0, len(video_files), batch_size), desc=f"Evaluating {model_name}"):
        batch_files = video_files[i:i+batch_size]
        videos = []
        class_names = []
        
        for video_path, class_name in batch_files:
            try:
                frames = load_video_frames(video_path)
                video_tensor = transform_video_frames(frames)
                videos.append(video_tensor)
                class_names.append(class_name)
            except Exception as e:
                print(f"Error loading {video_path}: {e}")
                continue
        
        if not videos:
            continue
        
        videos = paddle.stack(videos)
        
        with paddle.no_grad():
            outputs = model(videos)
        
        top5 = outputs.topk(5, axis=1)
        top5_pred = top5.indices.numpy()
        
        total += len(class_names)
        for idx, class_name in enumerate(class_names):
            class_name_normalized = class_name.replace("_", " ")
            top5_classes = [K400_CLASSES[i] if i < len(K400_CLASSES) else f"class_{i}" 
                            for i in top5_pred[idx]]
            
            if class_name_normalized in top5_classes[:1]:
                correct_top1 += 1
            if class_name_normalized in top5_classes:
                correct_top5 += 1
    
    if total == 0:
        return None
    
    return {
        "model": model_name,
        "type": "video",
        "top1": 100.0 * correct_top1 / total,
        "top5": 100.0 * correct_top5 / total,
        "total": total,
        "target": target_acc
    }

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate all Hiera models (image + video)")
    parser.add_argument("--image_data_dir", type=str, default="imagenet_official/imagenet-val",
                        help="ImageNet validation data directory")
    parser.add_argument("--video_data_dir", type=str, default="./video_test_data/train",
                        help="Video test data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="gpu", help="Device (gpu/cpu)")
    parser.add_argument("--models", type=str, default="all",
                        help="Models to evaluate: 'all', 'image', 'video', or comma-separated list")
    parser.add_argument("--num_images", type=int, default=0, 
                        help="Number of images to evaluate (0 = all)")
    parser.add_argument("--num_videos", type=int, default=0,
                        help="Number of videos to evaluate (0 = all)")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Hiera Unified Model Evaluation")
    print("=" * 70)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    if args.models == "all":
        image_models = list(IMAGE_MODEL_CONFIGS.keys())
        video_models = list(VIDEO_MODEL_CONFIGS.keys())
    elif args.models == "image":
        image_models = list(IMAGE_MODEL_CONFIGS.keys())
        video_models = []
    elif args.models == "video":
        image_models = []
        video_models = list(VIDEO_MODEL_CONFIGS.keys())
    else:
        selected = [m.strip() for m in args.models.split(",")]
        image_models = [m for m in selected if m in IMAGE_MODEL_CONFIGS]
        video_models = [m for m in selected if m in VIDEO_MODEL_CONFIGS]
    
    print(f"Image models: {image_models if image_models else 'None'}")
    print(f"Video models: {video_models if video_models else 'None'}")
    
    image_results = []
    video_results = []
    
    for model_name in image_models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print("=" * 70)
        try:
            result = evaluate_image_model(
                model_name, args.image_data_dir, args.batch_size,
                args.device, args.num_images
            )
            if result:
                image_results.append(result)
                print(f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    for model_name in video_models:
        print(f"\n{'=' * 70}")
        print(f"Evaluating {model_name}")
        print("=" * 70)
        try:
            result = evaluate_video_model(
                model_name, args.video_data_dir, args.batch_size,
                args.device, args.num_videos
            )
            if result:
                video_results.append(result)
                print(f"\n{model_name}: Top-1={result['top1']:.2f}%, Top-5={result['top5']:.2f}%")
        except Exception as e:
            print(f"\n{model_name}: Error - {e}")
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    
    if image_results:
        print("\n--- Image Models (ImageNet) ---")
        print(f"{'Model':<25} {'Top-1':>8} {'Top-5':>8} {'Target':>8} {'Images':>8}")
        print("-" * 70)
        for r in image_results:
            print(f"{r['model']:<25} {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['target']:>8.1f}% {r['total']:>8}")
    
    if video_results:
        print("\n--- Video Models (Kinetics-400) ---")
        print(f"{'Model':<25} {'Top-1':>8} {'Top-5':>8} {'Target':>8} {'Videos':>8}")
        print("-" * 70)
        for r in video_results:
            print(f"{r['model']:<25} {r['top1']:>8.2f}% {r['top5']:>8.2f}% {r['target']:>8.1f}% {r['total']:>8}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
