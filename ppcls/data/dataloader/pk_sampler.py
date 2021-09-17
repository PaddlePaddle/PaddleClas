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
from collections import defaultdict
import numpy as np
import random
from paddle.io import DistributedBatchSampler

from ppcls.utils import logger


class PKSampler(DistributedBatchSampler):
    """
    First, randomly sample P identities.
    Then for each identity randomly sample K instances.
    Therefore batch size is P*K, and the sampler called PKSampler.
    Args:
        dataset (paddle.io.Dataset): list of (img_path, pid, cam_id).
        sample_per_id(int): number of instances per identity in a batch.
        batch_size (int): number of examples in a batch.
        shuffle(bool): whether to shuffle indices order before generating
            batch indices. Default False.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 sample_per_id,
                 shuffle=True,
                 drop_last=True):
        super(PKSampler, self).__init__(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        assert batch_size % sample_per_id == 0, \
            "PKSampler configs error, Sample_per_id must be a divisor of batch_size."
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.sample_per_id = sample_per_id
        self.label_dict = defaultdict(list)
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        self.id_list = list(self.label_dict)

    def __iter__(self):
        if self.shuffle:
            np.random.RandomState(self.epoch).shuffle(self.id_list)
        id_list = self.id_list[self.local_rank * len(self):(self.local_rank + 1
                                                            ) * len(self)]
        id_per_batch = self.batch_size / self.sample_per_id
        for i in range(len(self)):
            batch_index = []
            for label_id in id_list[i * id_per_batch:(i + 1) * id_per_batch]:
                idx_label_list = self.label_dict[label_id]
                if self.sample_per_id <= len(idx_label_list):
                    batch_index.extend(
                        np.random.choice(
                            idx_label_list,
                            size=self.sample_per_id,
                            replace=False))
                else:
                    batch_index.extend(
                        np.random.choice(
                            idx_label_list,
                            size=self.sample_per_id,
                            replace=True))
            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index
