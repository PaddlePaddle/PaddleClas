import math
import os
import pickle
from typing import Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import paddle

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False


def load_pytorch_weights(url: str) -> Dict:
    """Load PyTorch weights and convert to PaddlePaddle format."""
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to load PyTorch weights. Please install it with: pip install torch")
    
    import urllib.request
    
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
                state_dict = load_pytorch_weights(checkpoints[checkpoint])
                if "head.projection.weight" in state_dict:
                    head_weight = state_dict["head.projection.weight"]
                    num_classes_from_ckpt = head_weight.shape[0]
                    if "num_classes" not in kwdargs:
                        kwdargs["num_classes"] = num_classes_from_ckpt
                    elif kwdargs["num_classes"] != num_classes_from_ckpt:
                        del state_dict["head.projection.weight"]
                        del state_dict["head.projection.bias"]
            model = model_func(**kwdargs)
            if pretrained:
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
    """
    Returns a conv with nd (e.g., Conv2d for n=2). Work up to n=3.
    If you wanted a 4d Hiera, you could probably just implement this for n=4. (no promises)
    """
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
    """Zero-out the masked regions of the input before conv.
    Prevents leakage of masked regions when using overlapping kernels.
    """
    if conv is None:
        return x
    if mask is None:
        return conv(x)
    mask = get_resized_mask(target_size=x.shape[2:], mask=mask)
    return conv(x * mask.bool())


def undo_windowing(
    x: paddle.Tensor, shape: List[int], mu_shape: List[int]
) -> paddle.Tensor:
    """
    Restore spatial organization by undoing windowed organization of mask units.

    Args:
        x: organized by mask units windows, e.g. in 2d [B, #MUy*#MUx, MUy, MUx, C]
        shape: current spatial shape, if it were not organized into mask unit
            windows, e.g. in 2d [B, #MUy*MUy, #MUx*MUx, C].
        mu_shape: current mask unit shape, e.g. in 2d [MUy, MUx]
    Returns:
        x: e.g. in 2d, [B, #MUy*MUy, #MUx*MUx, C]
    """
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
    """
    Reorders the tokens such that patches are contiguous in memory.
    E.g., given [B, (H, W), C] and stride of (Sy, Sx), this will re-order the tokens as
                           [B, (Sy, Sx, H // Sy, W // Sx), C]

    This allows operations like Max2d to be computed as x.view(B, Sx*Sy, -1, C).max(dim=1).
    Not only is this faster, but it also makes it easy to support inputs of arbitrary
    dimensions in addition to patch-wise sparsity.

    Performing this operation multiple times in sequence puts entire windows as contiguous
    in memory. For instance, if you applied the stride (2, 2) 3 times, entire windows of
    size 8x8 would be contiguous in memory, allowing operations like mask unit attention
    computed easily and efficiently, while also allowing max to be applied sequentially.

    Note: This means that intermediate values of the model are not in HxW order, so they
    need to be re-rolled if you want to use the intermediate values as a HxW feature map.
    The last block of the network is fine though, since by then the strides are all consumed.
    """

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
        """
        Input: Flattened patch embeddings [B, N, C]
        Output: Patch embeddings [B, N, C] permuted such that [B, 4, N//4, C].max(1) etc. performs MaxPoolNd
        """
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
    """
    Undos the "unroll" operation so that you can use intermediate features.
    """

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
        """
        Roll the given tensor back up to spatial order assuming it's from the given block.

        If no mask is provided:
            - Returns [B, H, W, C] for 2d, [B, T, H, W, C] for 3d, etc.
        If a mask is provided:
            - Returns [B, #MUs, MUy, MUx, C] for 2d, etc.
        """
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
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

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
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

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
