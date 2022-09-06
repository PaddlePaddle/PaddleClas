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
from paddle.io import DistributedBatchSampler
from ppcls.utils import logger


class PKSampler(DistributedBatchSampler):
    """First, randomly sample P identities.
        Then for each identity randomly sample K instances.
        Therefore batch size equals to P * K, and the sampler called PKSampler.

    Args:
        dataset (Dataset): Dataset which contains list of (img_path, pid, camid))
        batch_size (int): batch size
        sample_per_id (int): number of instance(s) within an class
        shuffle (bool, optional): _description_. Defaults to True.
        id_list(list): list of (start_id, end_id, start_id, end_id) for set of ids to duplicated.
        ratio(list): list of (ratio1, ratio2..) the duplication number for ids in id_list.
        drop_last (bool, optional): whether to discard the data at the end. Defaults to True.
        sample_method (str, optional): sample method when generating prob_list. Defaults to "sample_avg_prob".
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 sample_per_id,
                 shuffle=True,
                 drop_last=True,
                 id_list=None,
                 ratio=None,
                 sample_method="sample_avg_prob"):
        super().__init__(
            dataset, batch_size, shuffle=shuffle, drop_last=drop_last)
        assert batch_size % sample_per_id == 0, \
            f"PKSampler configs error, sample_per_id({sample_per_id}) must be a divisor of batch_size({batch_size})."
        assert hasattr(self.dataset,
                       "labels"), "Dataset must have labels attribute."
        self.sample_per_label = sample_per_id
        self.label_dict = defaultdict(list)
        self.sample_method = sample_method
        for idx, label in enumerate(self.dataset.labels):
            self.label_dict[label].append(idx)
        self.label_list = list(self.label_dict)
        assert len(self.label_list) * self.sample_per_label > self.batch_size, \
            "batch size should be smaller than "
        if self.sample_method == "id_avg_prob":
            self.prob_list = np.array([1 / len(self.label_list)] *
                                      len(self.label_list))
        elif self.sample_method == "sample_avg_prob":
            counter = []
            for label_i in self.label_list:
                counter.append(len(self.label_dict[label_i]))
            self.prob_list = np.array(counter) / sum(counter)
        else:
            logger.error(
                "PKSampler only support id_avg_prob and sample_avg_prob sample method, "
                "but receive {}.".format(self.sample_method))

        if id_list and ratio:
            assert len(id_list) % 2 == 0 and len(id_list) == len(ratio) * 2
            for i in range(len(self.prob_list)):
                for j in range(len(ratio)):
                    if i >= id_list[j * 2] and i <= id_list[j * 2 + 1]:
                        self.prob_list[i] = self.prob_list[i] * ratio[j]
                        break
            self.prob_list = self.prob_list / sum(self.prob_list)

        diff = np.abs(sum(self.prob_list) - 1)
        if diff > 0.00000001:
            self.prob_list[-1] = 1 - sum(self.prob_list[:-1])
            if self.prob_list[-1] > 1 or self.prob_list[-1] < 0:
                logger.error("PKSampler prob list error")
            else:
                logger.info(
                    "PKSampler: sum of prob list not equal to 1, diff is {}, change the last prob".
                    format(diff))

    def __iter__(self):
        label_per_batch = self.batch_size // self.sample_per_label
        for _ in range(len(self)):
            batch_index = []
            batch_label_list = np.random.choice(
                self.label_list,
                size=label_per_batch,
                replace=False,
                p=self.prob_list)
            for label_i in batch_label_list:
                label_i_indexes = self.label_dict[label_i]
                if self.sample_per_label <= len(label_i_indexes):
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_label,
                            replace=False))
                else:
                    batch_index.extend(
                        np.random.choice(
                            label_i_indexes,
                            size=self.sample_per_label,
                            replace=True))
            if not self.drop_last or len(batch_index) == self.batch_size:
                yield batch_index
