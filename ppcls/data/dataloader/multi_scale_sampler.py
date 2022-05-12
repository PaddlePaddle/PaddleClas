from paddle.io import Sampler
import paddle.distributed as dist

import math
import random
import numpy as np

from ppcls import data


class MultiScaleSampler(Sampler):
    def __init__(self,
                 data_source,
                 scales,
                 first_bs,
                 divided_factor=32,
                 is_training=True,
                 seed=None):
        """
            multi scale samper
            Args:
                data_source(dataset)
                scales(list): several scales for image resolution
                first_bs(int): batch size for the first scale in scales
                divided_factor(int): ImageNet models down-sample images by a factor, ensure that width and height dimensions are multiples are multiple of devided_factor.
                is_training(boolean): mode 
        """
        # min. and max. spatial dimensions
        self.data_source = data_source
        self.n_data_samples = len(self.data_source)

        if isinstance(scales[0], tuple):
            width_dims = [i[0] for i in scales]
            height_dims = [i[1] for i in scales]
        elif isinstance(scales[0], int):
            width_dims = scales
            height_dims = scales
        base_im_w = width_dims[0]
        base_im_h = height_dims[0]
        base_batch_size = first_bs

        # Get the GPU and node related information
        num_replicas = dist.get_world_size()
        rank = dist.get_rank()
        # adjust the total samples to avoid batch dropping
        num_samples_per_replica = int(
            math.ceil(self.n_data_samples * 1.0 / num_replicas))
        img_indices = [idx for idx in range(self.n_data_samples)]

        self.shuffle = False
        if is_training:
            # compute the spatial dimensions and corresponding batch size
            # ImageNet models down-sample images by a factor of 32.
            # Ensure that width and height dimensions are multiples are multiple of 32.
            width_dims = [
                int((w // divided_factor) * divided_factor) for w in width_dims
            ]
            height_dims = [
                int((h // divided_factor) * divided_factor)
                for h in height_dims
            ]

            img_batch_pairs = list()
            base_elements = base_im_w * base_im_h * base_batch_size
            for (h, w) in zip(height_dims, width_dims):
                batch_size = int(max(1, (base_elements / (h * w))))
                img_batch_pairs.append((w, h, batch_size))
            self.img_batch_pairs = img_batch_pairs
            self.shuffle = True
        else:
            self.img_batch_pairs = [(base_im_w, base_im_h, base_batch_size)]

        self.img_indices = img_indices
        self.n_samples_per_replica = num_samples_per_replica
        self.epoch = 0
        self.rank = rank
        self.num_replicas = num_replicas
        self.seed = seed
        self.batch_list = []
        self.current = 0
        indices_rank_i = self.img_indices[self.rank:len(self.img_indices):
                                          self.num_replicas]
        while self.current < self.n_samples_per_replica:
            curr_w, curr_h, curr_bsz = random.choice(self.img_batch_pairs)

            end_index = min(self.current + curr_bsz,
                            self.n_samples_per_replica)

            batch_ids = indices_rank_i[self.current:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz:
                batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
            self.current += curr_bsz

            if len(batch_ids) > 0:
                batch = [curr_w, curr_h, len(batch_ids)]
                self.batch_list.append(batch)
        self.length = len(self.batch_list)

    def __iter__(self):
        if self.shuffle:
            if self.seed is not None:
                random.seed(self.seed)
            else:
                random.seed(self.epoch)
            random.shuffle(self.img_indices)
            random.shuffle(self.img_batch_pairs)
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):
                                              self.num_replicas]
        else:
            indices_rank_i = self.img_indices[self.rank:len(self.img_indices):
                                              self.num_replicas]

        start_index = 0
        for batch_tuple in self.batch_list:
            curr_w, curr_h, curr_bsz = batch_tuple
            end_index = min(start_index + curr_bsz, self.n_samples_per_replica)
            batch_ids = indices_rank_i[start_index:end_index]
            n_batch_samples = len(batch_ids)
            if n_batch_samples != curr_bsz:
                batch_ids += indices_rank_i[:(curr_bsz - n_batch_samples)]
            start_index += curr_bsz

            if len(batch_ids) > 0:
                batch = [(curr_w, curr_h, b_id) for b_id in batch_ids]
                yield batch

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self):
        return self.length
