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
import numpy as np
import math
from paddle.io import DistributedBatchSampler


class RepeatedAugSampler(DistributedBatchSampler):
    """
    It ensures that different each augmented version of a sample will be visible to a
    different process (GPU). Heavily based on torch.utils.data.DistributedSampler
    Code was based on https://github.com/facebookresearch/deit/blob/0c4b8f60/samplers.py
    and https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/distributed_sampler.py
    Args:
        dataset(paddle.io.Dataset): list of (img_path, pid, cam_id).
        batch_size(int): number of examples in a batch.
        num_replicas(optional): Number of processes participating in
            distributed training.
        rank(optional): Rank of the current process within num_replicas.
        shuffle(bool): whether to shuffle indices order before generating
            batch indices. Default False.
        drop_last(bool): whether drop the last incomplete batch dataset size
            is not divisible by the batch size. Default False
        num_repeats(int): number of repeat instances in a batch.
        selected_round(int): number of smallest divisible number.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 num_repeats=3,
                 selected_round=256):
        super().__init__(dataset, batch_size, num_replicas, rank, shuffle,
                         drop_last)
        self.num_repeats = num_repeats
        self.num_samples = int(
            math.ceil(len(self.dataset) * num_repeats / self.nranks))
        self.total_size = self.num_samples * self.nranks
        self.num_selected_samples = int(
            math.floor(
                len(self.dataset) // selected_round * selected_round /
                self.nranks))

    def __iter__(self):
        num_samples = len(self.dataset)
        indices = np.arange(num_samples).tolist()
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(indices)
            self.epoch += 1
        indices = [ele for ele in indices for i in range(self.num_repeats)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.local_rank:self.total_size:self.nranks]
        assert len(indices) == self.num_samples
        _sample_iter = iter(indices[:self.num_selected_samples])

        batch_indices = []
        for idx in _sample_iter:
            batch_indices.append(idx)
            if len(batch_indices) == self.batch_size:
                yield batch_indices
                batch_indices = []
        if not self.drop_last and len(batch_indices) > 0:
            yield batch_indices

    def __len__(self):
        num_samples = self.num_selected_samples
        num_samples += int(not self.drop_last) * (self.batch_size - 1)
        return num_samples // self.batch_size
