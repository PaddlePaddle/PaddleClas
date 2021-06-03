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
import copy
import paddle
import numpy as np
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader

from ppcls.utils import logger

from . import dataset
from . import imaug
from . import samplers
# dataset
from .dataset.imagenet_dataset import ImageNetDataset
from .dataset.multilabel_dataset import MultiLabelDataset
from .dataset.common_dataset import create_operators
from .dataset.vehicle_dataset import CompCars, VeriWild

# sampler
from .samplers import DistributedRandomIdentitySampler

from .preprocess import transform


def build_dataloader(config, mode, device, seed=None):
    assert mode in ['Train', 'Eval', 'Test'
                    ], "Mode should be Train, Eval or Test."
    # build dataset
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    if 'batch_transform_ops' in config_dataset:
        batch_transform = config_dataset.pop('batch_transform_ops')
    else:
        batch_transform = None

    dataset = eval(dataset_name)(**config_dataset)

    logger.info("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]['sampler']
    if "name" not in config_sampler:
        batch_sampler = None
        batch_size = config_sampler["batch_size"]
        drop_last = config_sampler["drop_last"]
        shuffle = config_sampler["shuffle"]
    else:
        sampler_name = config_sampler.pop("name")
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)

    logger.info("build batch_sampler({}) success...".format(batch_sampler))

    # build batch operator
    def mix_collate_fn(batch):
        batch = transform(batch, batch_ops)
        # batch each field
        slots = []
        for items in batch:
            for i, item in enumerate(items):
                if len(slots) < len(items):
                    slots.append([item])
                else:
                    slots[i].append(item)
        return [np.stack(slot, axis=0) for slot in slots]

    if isinstance(batch_transform, list):
        batch_ops = create_operators(batch_transform)
        batch_collate_fn = mix_collate_fn
    else:
        batch_collate_fn = None

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=batch_collate_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            places=device,
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_sampler=batch_sampler,
            collate_fn=batch_collate_fn)

    logger.info("build data_loader({}) success...".format(data_loader))

    return data_loader


'''
# TODO: fix the format
def build_dataloader(config, mode, device, seed=None):
    from . import reader
    from .reader import Reader
    dataloader = Reader(config, mode=mode, places=device)()
    return dataloader

'''
