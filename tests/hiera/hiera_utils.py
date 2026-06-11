import math
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import paddle

PADDLE_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "weights")

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
    
    return None


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
                
                if state_dict is None:
                    state_dict = load_pytorch_weights(checkpoints[checkpoint])
                
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
