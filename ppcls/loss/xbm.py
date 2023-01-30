# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Tuple

import paddle


class CrossBatchMemory(paddle.nn.Layer):
    """
    CrossBatchMemory Implementation. refer to "Cross-Batch Memory for Embedding Learning".

    code heavily based on https://github.com/msight-tech/research-xbm/blob/master/ret_benchmark/modeling/xbm.py

    Args:
        size (int): Size of memory bank
        embedding_size (int): number of embedding dimension for memory bank
    """

    def __init__(self, size: int, embedding_size: int):
        super().__init__()
        self.size = size
        self.embedding_size = embedding_size

        # initialize and register feature queue for resume training
        feats = paddle.zeros([self.size, self.embedding_size])
        self.register_buffer("feats", feats)

        # initialize and register label queue for resume training
        targets = paddle.zeros([self.size, ], dtype="int64")
        self.register_buffer("targets", targets)

        self.ptr = 0
        # self.accumulated_size = 0

    @property
    def _is_full(self) -> bool:
        # return self.accumulated_size >= self.size
        return self.targets[-1].item() != 0  # author's usage

    def get(self) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """return features and targets in memory bank

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: [features, targets]
        """
        if self._is_full:
            return self.feats, self.targets
        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]

    def enqueue_dequeue(self, feats: paddle.Tensor,
                        targets: paddle.Tensor) -> None:
        """put newest feats and targets into memory bank and pop oldest feats and targets from momory bank

        Args:
            feats (paddle.Tensor): features to enque
            targets (paddle.Tensor): targets to enque
        """
        input_size = len(targets)
        if self.ptr + input_size > self.size:
            self.feats[-input_size:] = feats
            self.targets[-input_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr:self.ptr + input_size] = feats
            self.targets[self.ptr:self.ptr + input_size] = targets
            self.ptr += input_size
        # self.accumulated_size += input_size

    def forward(self, *kargs, **kwargs):
        raise NotImplementedError(
            "CrossBatchMemory module is for memory-bank, forward method is not needed"
        )
