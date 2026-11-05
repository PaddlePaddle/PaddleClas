import paddle
import inspect
import math
from functools import partial, wraps
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


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


PADDLE_WEIGHTS_MAP = {
    "hiera_tiny_224": {
        "mae_in1k_ft_in1k": "hiera_tiny_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_small_224": {
        "mae_in1k_ft_in1k": "hiera_small_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_base_224": {
        "mae_in1k_ft_in1k": "hiera_base_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_base_plus_224": {
        "mae_in1k_ft_in1k": "hiera_base_plus_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_large_224": {
        "mae_in1k_ft_in1k": "hiera_large_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_huge_224": {
        "mae_in1k_ft_in1k": "hiera_huge_224_mae_in1k_ft_in1k.pdparams",
    },
    "hiera_base_16x224": {
        "mae_k400_ft_k400": "hiera_base_16x224_mae_k400_ft_k400.pdparams",
    },
    "hiera_base_plus_16x224": {
        "mae_k400_ft_k400": "hiera_base_plus_16x224_mae_k400_ft_k400.pdparams",
    },
    "hiera_large_16x224": {
        "mae_k400_ft_k400": "hiera_large_16x224_mae_k400_ft_k400.pdparams",
    },
    "hiera_huge_16x224": {
        "mae_k400_ft_k400": "hiera_huge_16x224_mae_k400_ft_k400.pdparams",
    },
}


MODEL_URLS = {
    "hiera_tiny_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_tiny_224_mae_in1k_ft_in1k.pdparams",
    "hiera_small_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_small_224_mae_in1k_ft_in1k.pdparams",
    "hiera_base_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_224_mae_in1k_ft_in1k.pdparams",
    "hiera_base_plus_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_plus_224_mae_in1k_ft_in1k.pdparams",
    "hiera_large_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_large_224_mae_in1k_ft_in1k.pdparams",
    "hiera_huge_224_mae_in1k_ft_in1k.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_huge_224_mae_in1k_ft_in1k.pdparams",
    "hiera_base_16x224_mae_k400_ft_k400.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_16x224_mae_k400_ft_k400.pdparams",
    "hiera_base_plus_16x224_mae_k400_ft_k400.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_plus_16x224_mae_k400_ft_k400.pdparams",
    "hiera_large_16x224_mae_k400_ft_k400.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_large_16x224_mae_k400_ft_k400.pdparams",
    "hiera_huge_16x224_mae_k400_ft_k400.pdparams":
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_huge_16x224_mae_k400_ft_k400.pdparams",
}


def load_paddle_weights(model_name: str, checkpoint: str) -> Dict:
    """Load Paddle weights, downloading and caching them from MODEL_URLS if needed."""
    if model_name not in PADDLE_WEIGHTS_MAP:
        return None

    checkpoint_map = PADDLE_WEIGHTS_MAP[model_name]
    if checkpoint not in checkpoint_map:
        return None

    weight_file = checkpoint_map[checkpoint]
    url = MODEL_URLS.get(weight_file)
    if url is None:
        raise FileNotFoundError(
            f"No pretrained weights available for: {weight_file}")

    from paddle.utils.download import get_weights_path_from_url
    weight_path = get_weights_path_from_url(url)

    print(f"Loading Paddle weights from: {weight_path}")
    return paddle.load(weight_path)


def pretrained_model(checkpoints: Dict[str, str], default: str = None) -> Callable:
    """Loads a Hiera model from a pretrained source (if pretrained=True). Use "checkpoint" to specify the checkpoint."""

    def inner(model_func: Callable) -> Callable:
        model_name = model_func.__name__
        
        @wraps(model_func)
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
                    ckpt_class_num = head_weight.shape[0]
                    if "class_num" not in kwdargs:
                        kwdargs["class_num"] = ckpt_class_num
                    elif kwdargs["class_num"] != ckpt_class_num:
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
        class_num: int,
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
        self.projection = paddle.compat.nn.Linear(dim, class_num)
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


class Hiera(paddle.nn.Module):
    @has_config
    def __init__(
        self,
        input_size: Tuple[int, ...] = (224, 224),
        in_chans: int = 3,
        embed_dim: int = 96,
        num_heads: int = 1,
        class_num: int = 1000,
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
        self.head = Head(embed_dim, class_num, dropout_rate=head_dropout)
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
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_tiny_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_tiny_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_small_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_small_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(1, 2, 11, 2), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_224(**kwdargs):
    return Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_plus_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_base_plus_224(**kwdargs):
    return Hiera(embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_large_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_large_224(**kwdargs):
    return Hiera(embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_in1k_ft_in1k": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_huge_224_mae_in1k_ft_in1k.pdparams",
    },
    default="mae_in1k_ft_in1k",
)
def hiera_huge_224(**kwdargs):
    return Hiera(embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs)


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_16x224_mae_k400_ft_k400.pdparams",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_16x224(class_num: int = 400, **kwdargs):
    return Hiera(
        class_num=class_num,
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
        "mae_k400_ft_k400": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_base_plus_16x224_mae_k400_ft_k400.pdparams",
    },
    default="mae_k400_ft_k400",
)
def hiera_base_plus_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=112, num_heads=2, stages=(2, 3, 16, 3), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_large_16x224_mae_k400_ft_k400.pdparams",
    },
    default="mae_k400_ft_k400",
)
def hiera_large_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=144, num_heads=2, stages=(2, 6, 36, 4), **kwdargs
    )


@pretrained_model(
    {
        "mae_k400_ft_k400": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/hiera_huge_16x224_mae_k400_ft_k400.pdparams",
    },
    default="mae_k400_ft_k400",
)
def hiera_huge_16x224(**kwdargs):
    return hiera_base_16x224(
        embed_dim=256, num_heads=4, stages=(2, 6, 36, 4), **kwdargs
    )
