#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

# reference: https://arxiv.org/abs/2110.02178

import math
import random

import numpy as np
import paddle
from paddle.io import Sampler
import paddle.distributed as dist

from ppcls.utils import logger


class MultiScaleSampler(Sampler):
    def __init__(self,
                 dataset,
                 scales,
                 first_bs,
                 divided_factor=32,
                 shuffle=True,
                 drop_last=False,
                 seed=None):
        """Multi-Scale Sampler

        Args:
            dataset (paddle.io.Dataset):
            scales (list): List of scales for image resolution.
            first_bs (int): batch size for the first scale in 'scales'.
            divided_factor (int, optional): Ensure that width and height are multiple of 'devided_factor'. Defaults to 32.
            shuffle (bool, optional):
            drop_last (bool, optional):
            seed (int, optional):

        Raises:
            RuntimeError: The type of element of 'scales' list is not one of 'int', 'tuple' or 'list'.
        """
        self.seed = seed
        self.shuffle = shuffle
        self.num_samples = len(dataset)

        if isinstance(scales[0], (tuple, list)):
            width_dims = [(i[0] // divided_factor) * divided_factor
                          for i in scales]
            height_dims = [(i[1] // divided_factor) * divided_factor
                           for i in scales]
        elif isinstance(scales[0], int):
            width_dims = [(i // divided_factor) * divided_factor
                          for i in scales]
            height_dims = width_dims
        else:
            msg = "The element of 'scales' must be int, tuple or list, and length is 2 when tuple and list."
            logger.error(msg)
            raise RuntimeError(msg)

        element_num = width_dims[0] * height_dims[0] * first_bs
        self.batch_infos = []
        for width, height in zip(width_dims, height_dims):
            batch_size = int(max(1, (element_num / (width * height))))
            self.batch_infos.append((width, height, batch_size))

        self.num_replicas = dist.get_world_size()
        self.rank = dist.get_rank()
        self.num_samples_per_replica = int(
            math.ceil(self.num_samples * 1.0 / self.num_replicas))
        self.total_size = self.num_samples_per_replica * self.num_replicas

        self.batch_info_list = []
        count = 0
        while count < self.num_samples_per_replica:
            # TODO(gaotingquan): this random seems to have failed
            width, height, batch_size = random.choice(self.batch_infos)
            count += batch_size
            self.batch_info_list.append([width, height, batch_size])
        if count > self.num_samples_per_replica:
            if not drop_last:
                # TODO(gaotingquan): inconsistent with ml-cvnet(Apple)
                self.batch_info_list[-1][
                    2] = self.num_samples_per_replica - count + batch_size
            else:
                self.batch_info_list.pop(-1)

        self.length = len(self.batch_info_list)
        self.epoch = 0

    def __iter__(self):
        indices = np.arange(self.num_samples).tolist()

        if self.shuffle:
            if self.seed:
                random.seed(self.seed)
                random.shuffle(indices)
                random.seed(self.seed)
                random.shuffle(self.batch_info_list)
            else:
                np.random.RandomState(self.epoch).shuffle(indices)
                self.epoch += 1
                random.shuffle(indices)
                random.shuffle(self.batch_info_list)

        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        indices = indices[self.rank:len(indices):self.num_replicas]

        idx = 0
        for width, height, batch_size in self.batch_info_list:
            batch_indices = indices[idx:idx + batch_size]
            idx += batch_size
            yield [(width, height, img_idx) for img_idx in batch_indices]

    def __len__(self):
        return self.length

    def set_epoch(self, epoch):
        self.epoch = epoch
