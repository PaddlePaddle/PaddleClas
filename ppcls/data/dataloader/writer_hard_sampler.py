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
import copy
import random
from paddle.io import DistributedBatchSampler

from ppcls.data.dataloader.writer_hard_dataset import WriterHardDataset


class WriterHardSampler(DistributedBatchSampler):
    """
    Randomly sample N anchor, then for each identity,
    randomly sample 2 positive and 1 negative for each anchor, therefore batch size is N*4.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, dataset, batch_size, shuffle=True, **args):
        super(WriterHardSampler, self).__init__(dataset, batch_size)
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        assert not self.batch_size % 4, "bs of WriterHardSampler should be 3*N"
        assert isinstance(dataset, WriterHardDataset), "WriterHardSampler only support WriterHardDataset"
        self.num_pids_per_batch = self.batch_size // 4
        self.anchor_list = []
        self.person_id_map = {}
        self.text_id_map = {}
        anno_list = dataset.anno_list
        for i, anno_i in enumerate(anno_list):
            _, person_id, text_id = anno_i.strip().split(" ")
            if text_id != "-1":
                if random.random() < 0.5:
                    self.anchor_list.append([i, person_id, text_id])
                else:
                    if text_id in self.text_id_map:
                        self.text_id_map[text_id].append(i)
                    else:
                        self.text_id_map[text_id] = [i]
            else:
                if person_id in self.person_id_map:
                    self.person_id_map[person_id].append(i)
                else:
                    self.person_id_map[person_id] = [i]
        assert len(self.anchor_list) > self.batch_size, "anchor should be larger than batch_size"

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.anchor_list)
        for i in range(len(self)):
            batch_indices = []
            for j in range(self.batch_size // 4):
                anchor = self.anchor_list[i * self.batch_size // 4 + j]
                anchor_index = anchor[0]
                anchor_person_id = anchor[1]
                anchor_text_id = anchor[2]
                person_indices = random.sample(self.person_id_map[anchor_person_id], 2)
                text_index = random.choice(self.text_id_map[anchor_text_id])
                batch_indices.append(anchor_index)
                batch_indices += person_indices
                batch_indices.append(text_index)
            yield batch_indices

    def __len__(self):
        return len(self.anchor_list) * 4 // self.batch_size
