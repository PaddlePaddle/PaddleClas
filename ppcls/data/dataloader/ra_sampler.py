import math
import numpy as np

from paddle.io import DistributedBatchSampler


class RASampler(DistributedBatchSampler):
    """
    based on https://github.com/facebookresearch/deit/blob/main/samplers.py
    """

    def __init__(self,
                 dataset,
                 batch_size,
                 num_replicas=None,
                 rank=None,
                 shuffle=False,
                 drop_last=False,
                 num_repeats: int=3):
        super().__init__(dataset, batch_size, num_replicas, rank, shuffle,
                         drop_last)
        self.num_repeats = num_repeats
        self.num_samples = int(
            math.ceil(len(self.dataset) * num_repeats / self.nranks))
        self.total_size = self.num_samples * self.nranks
        self.num_selected_samples = int(
            math.floor(len(self.dataset) // 256 * 256 / self.nranks))

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
