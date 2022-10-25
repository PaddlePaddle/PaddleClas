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

import copy
import random
from collections import defaultdict

import numpy as np
from paddle.io import DistributedBatchSampler


class DistributedRandomIdentitySampler(DistributedBatchSampler):
    """Randomly sample N identities, then for each identity,
       randomly sample K instances, therefore batch size equals to N * K.
    Args:
        dataset(Dataset): Dataset which contains list of (img_path, pid, camid))
        batch_size (int): batch size
        num_instances (int): number of instance(s) within an class
        drop_last (bool): whether to discard the data at the end
        max_iters (int): max iteration(s). Default to None.
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances,
                 drop_last,
                 max_iters=None,
                 **args):
        assert batch_size % num_instances == 0, \
            f"batch_size({batch_size}) must be divisible by num_instances({num_instances}) when using DistributedRandomIdentitySampler"
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.drop_last = drop_last
        self.max_iters = max_iters
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, pid in enumerate(self.dataset.labels):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def _prepare_batch(self):
        batch_idxs_dict = defaultdict(list)
        count = []
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        count = [len(batch_idxs_dict[pid]) for pid in self.pids]
        count = np.array(count)
        avai_pids = copy.deepcopy(self.pids)
        return batch_idxs_dict, avai_pids, count

    def __iter__(self):
        # prepare
        batch_idxs_dict, avai_pids, count = self._prepare_batch()

        # sample
        if self.max_iters is not None:
            for _ in range(self.max_iters):
                final_idxs = []
                if len(avai_pids) < self.num_pids_per_batch:
                    batch_idxs_dict, avai_pids, count = self._prepare_batch()

                selected_pids = np.random.choice(avai_pids,
                                                 self.num_pids_per_batch,
                                                 False, count / count.sum())
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    pid_idx = avai_pids.index(pid)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.pop(pid_idx)
                        count = np.delete(count, pid_idx)
                    else:
                        count[pid_idx] = len(batch_idxs_dict[pid])
                yield final_idxs
        else:
            final_idxs = []
            while len(avai_pids) >= self.num_pids_per_batch:
                selected_pids = random.sample(avai_pids,
                                              self.num_pids_per_batch)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
            _sample_iter = iter(final_idxs)
            batch_indices = []
            for idx in _sample_iter:
                batch_indices.append(idx)
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
                    batch_indices = []
            if not self.drop_last and len(batch_indices) > 0:
                yield batch_indices

    def __len__(self):
        if self.max_iters is not None:
            return self.max_iters
        elif self.drop_last:
            return self.length // self.batch_size
        else:
            return (self.length + self.batch_size - 1) // self.batch_size
