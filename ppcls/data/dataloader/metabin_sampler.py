#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import copy
import itertools
from collections import defaultdict
import numpy as np
from paddle.io import Sampler, BatchSampler


class DomainShuffleSampler(Sampler):
    """
    Domain shuffle sampler
    Args:
        dataset(Dataset): Dataset for sampling
        batch_size (int): Number of examples in a batch.
        num_instances (int): Number of instances per identity in a batch.
        camera_to_domain (bool): If True, consider each camera as an individual domain
    
    Code was heavily based on https://github.com/bismex/MetaBIN
    reference: https://arxiv.org/abs/2011.14670v2
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_instances,
                 camera_to_domain=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_domain = defaultdict(list)
        self.pid_index = defaultdict(list)
        # data_source: [(img_path, pid, camera, domain), ...]  (camera_to_domain = True)
        if camera_to_domain:
            data_source = zip(dataset.images, dataset.labels, dataset.cameras,
                              dataset.cameras)
        else:
            data_source = zip(dataset.images, dataset.labels, dataset.cameras,
                              dataset.domains)
        for index, info in enumerate(data_source):
            domainid = info[3]
            if camera_to_domain:
                pid = 'p' + str(info[1]) + '_d' + str(domainid)
            else:
                pid = 'p' + str(info[1])
            self.index_pid[index] = pid
            self.pid_domain[pid] = domainid
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.domains = list(self.pid_domain.values())

        self.num_identities = len(self.pids)
        self.num_domains = len(set(self.domains))

        self.batch_size //= self.num_domains
        self.num_pids_per_batch //= self.num_domains

        val_pid_index = [len(x) for x in self.pid_index.values()]

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                val_pid_index_upper.append(x - v_remain + self.num_instances)

        cnt_domains = [0 for x in range(self.num_domains)]
        for val, index in zip(val_pid_index_upper, self.domains):
            cnt_domains[index] += val
        self.max_cnt_domains = max(cnt_domains)
        self.total_images = self.num_domains * (
            self.max_cnt_domains -
            (self.max_cnt_domains % self.batch_size) - self.batch_size)

    def _get_epoch_indices(self):
        def _get_batch_idxs(pids, pid_index, num_instances):
            batch_idxs_dict = defaultdict(list)
            for pid in pids:
                idxs = copy.deepcopy(pid_index[pid])
                if len(
                        idxs
                ) < self.num_instances:  # if idxs is smaller than num_instance, choice redundantly
                    idxs = np.random.choice(
                        idxs, size=self.num_instances, replace=True)
                elif (len(idxs) % self.num_instances) != 0:
                    idxs.extend(
                        np.random.choice(
                            idxs,
                            size=self.num_instances - len(idxs) %
                            self.num_instances,
                            replace=False))

                np.random.shuffle(idxs)
                batch_idxs = []
                for idx in idxs:
                    batch_idxs.append(int(idx))
                    if len(batch_idxs) == num_instances:
                        batch_idxs_dict[pid].append(batch_idxs)
                        batch_idxs = []
            return batch_idxs_dict

        batch_idxs_dict = _get_batch_idxs(self.pids, self.pid_index,
                                          self.num_instances)

        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)

        local_avai_pids = \
            [[pids for pids, idx in zip(avai_pids, self.domains) if idx == i]
             for i in list(set(self.domains))]
        local_avai_pids_save = copy.deepcopy(local_avai_pids)

        revive_idx = [False for i in range(self.num_domains)]
        final_idxs = []
        while len(avai_pids) >= self.num_pids_per_batch and not all(
                revive_idx):
            for i in range(self.num_domains):
                selected_pids = np.random.choice(
                    local_avai_pids[i], self.num_pids_per_batch, replace=False)
                for pid in selected_pids:
                    batch_idxs = batch_idxs_dict[pid].pop(0)
                    final_idxs.extend(batch_idxs)
                    if len(batch_idxs_dict[pid]) == 0:
                        avai_pids.remove(pid)
                        local_avai_pids[i].remove(pid)
            for i in range(self.num_domains):
                if len(local_avai_pids[i]) < self.num_pids_per_batch:
                    batch_idxs_dict_new = _get_batch_idxs(
                        self.pids, self.pid_index, self.num_instances)

                    revive_idx[i] = True
                    cnt = 0
                    for pid, val in batch_idxs_dict_new.items():
                        if self.domains[cnt] == i:
                            batch_idxs_dict[pid] = copy.deepcopy(
                                batch_idxs_dict_new[pid])
                        cnt += 1
                    local_avai_pids[i] = copy.deepcopy(local_avai_pids_save[i])
                    avai_pids.extend(local_avai_pids_save[i])
                    avai_pids = list(set(avai_pids))
        return final_idxs

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        while True:
            indices = self._get_epoch_indices()
            yield from indices


class DomainShuffleBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, num_instances, camera_to_domain,
                 drop_last):
        sampler = DomainShuffleSampler(
            dataset=dataset,
            batch_size=batch_size,
            num_instances=num_instances,
            camera_to_domain=camera_to_domain)
        super().__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last)


class NaiveIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
        dataset(Dataset): Dataset for sampling
        batch_size (int): Number of examples in a batch.
        num_instances (int): Number of instances per identity in a batch.

    Code was heavily based on https://github.com/bismex/MetaBIN
    reference: https://arxiv.org/abs/2011.14670v2
    """

    def __init__(self, dataset, batch_size, num_instances):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = batch_size // self.num_instances

        self.index_pid = defaultdict(list)
        self.pid_cam = defaultdict(list)
        self.pid_index = defaultdict(list)

        # data_source: [(img_path, pid, camera, domain), ...]  (camera_to_domain = True)
        data_source = zip(dataset.images, dataset.labels, dataset.cameras,
                          dataset.cameras)
        for index, info in enumerate(data_source):
            pid = info[1]
            camid = info[2]
            self.index_pid[index] = pid
            self.pid_cam[pid].append(camid)
            self.pid_index[pid].append(index)

        self.pids = list(self.pid_index.keys())
        self.num_identities = len(self.pids)

        val_pid_index = [len(x) for x in self.pid_index.values()]

        val_pid_index_upper = []
        for x in val_pid_index:
            v_remain = x % self.num_instances
            if v_remain == 0:
                val_pid_index_upper.append(x)
            else:
                val_pid_index_upper.append(x - v_remain + self.num_instances)

        total_images = sum(val_pid_index_upper)
        total_images = total_images - (total_images % self.batch_size
                                       ) - self.batch_size  # approax
        self.total_images = total_images

    def _get_epoch_indices(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(
                self.pid_index[pid])  # whole index for each ID
            if len(
                    idxs
            ) < self.num_instances:  # if idxs is smaller than num_instance, choice redundantly
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True)
            elif (len(idxs) % self.num_instances) != 0:
                idxs.extend(
                    np.random.choice(
                        idxs,
                        size=self.num_instances - len(idxs) %
                        self.num_instances,
                        replace=False))

            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(int(idx))
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # batch_idxs_dict: dictionary, len(batch_idxs_dict) is len(pidx), each pidx, num_instance x k samples
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids, self.num_pids_per_batch, replace=False)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0: avai_pids.remove(pid)
        return final_idxs

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        while True:
            indices = self._get_epoch_indices()
            yield from indices

    def __len__(self):
        return self.total_images


class NaiveIdentityBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, num_instances, drop_last):
        sampler = NaiveIdentitySampler(
            dataset=dataset,
            batch_size=batch_size,
            num_instances=num_instances)
        super().__init__(
            sampler=sampler, batch_size=batch_size, drop_last=drop_last)
