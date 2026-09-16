# -*- coding: utf-8 -*-
from typing import List, Optional

import paddle
import paddle.nn as nn
from paddle import Tensor

from mobileclip import logger


class GlobalPool(nn.Layer):
    pool_types = ["mean", "rms", "abs"]

    def __init__(
        self,
        pool_type: Optional[str] = "mean",
        keep_dim: Optional[bool] = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        self.pool_type = pool_type
        self.keep_dim = keep_dim

    def _global_pool(self, x: Tensor, dims: List):
        if self.pool_type == "rms":
            x = x**2
            x = paddle.mean(x, axis=dims, keepdim=self.keep_dim)
            x = x**-0.5
        elif self.pool_type == "abs":
            x = paddle.mean(paddle.abs(x), axis=dims, keepdim=self.keep_dim)
        else:
            x = paddle.mean(x, axis=dims, keepdim=self.keep_dim)
        return x

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            dims = [-2, -1]
        elif x.ndim == 5:
            dims = [-3, -2, -1]
        else:
            raise NotImplementedError("Currently 2D and 3D global pooling supported")
        return self._global_pool(x, dims=dims)


class GlobalPool2D(nn.Layer):
    def __init__(self, in_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        scale = in_dim**-0.5
        self.pool = GlobalPool(pool_type="mean", keep_dim=False)
        self.proj = self.create_parameter(
            shape=[in_dim, out_dim],
            default_initializer=nn.initializer.Normal(std=scale)
        )
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        assert (
            x.ndim == 4
        ), "Input should be 4-dimensional. Got: {}".format(x.shape)
        x = self.pool(x)
        x = paddle.matmul(x, self.proj)
        return x
