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

import inspect
import copy
import random
import platform

import paddle
import numpy as np
import paddle.distributed as dist
from functools import partial
from paddle.io import DistributedBatchSampler, BatchSampler, DataLoader
from ppcls.utils import logger

from ppcls.data import dataloader
# dataset
from ppcls.data.dataloader.imagenet_dataset import ImageNetDataset
from ppcls.data.dataloader.multilabel_dataset import MultiLabelDataset
from ppcls.data.dataloader.common_dataset import create_operators
from ppcls.data.dataloader.vehicle_dataset import CompCars, VeriWild
from ppcls.data.dataloader.logo_dataset import LogoDataset
from ppcls.data.dataloader.icartoon_dataset import ICartoonDataset
from ppcls.data.dataloader.mix_dataset import MixDataset
from ppcls.data.dataloader.multi_scale_dataset import MultiScaleDataset
from ppcls.data.dataloader.person_dataset import Market1501, MSMT17, DukeMTMC
from ppcls.data.dataloader.face_dataset import FiveValidationDataset, AdaFaceDataset
from ppcls.data.dataloader.custom_label_dataset import CustomLabelDataset
from ppcls.data.dataloader.cifar import Cifar10, Cifar100
from ppcls.data.dataloader.metabin_sampler import DomainShuffleBatchSampler, NaiveIdentityBatchSampler

# sampler
from ppcls.data.dataloader.DistributedRandomIdentitySampler import DistributedRandomIdentitySampler
from ppcls.data.dataloader.pk_sampler import PKSampler
from ppcls.data.dataloader.mix_sampler import MixSampler
from ppcls.data.dataloader.multi_scale_sampler import MultiScaleSampler
from ppcls.data import preprocess
from ppcls.data.preprocess import transform


def create_operators(params, class_num=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op_func = getattr(preprocess, op_name)
        if "class_num" in inspect.getfullargspec(op_func).args:
            param.update({"class_num": class_num})
        op = op_func(**param)
        ops.append(op)

    return ops


def worker_init_fn(worker_id: int, num_workers: int, rank: int, seed: int):
    """callback function on each worker subprocess after seeding and before data loading.

    Args:
        worker_id (int): Worker id in [0, num_workers - 1]
        num_workers (int): Number of subprocesses to use for data loading.
        rank (int): Rank of process in distributed environment. If in non-distributed environment, it is a constant number `0`.
        seed (int): Random seed
    """
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build(config, mode, use_dali=False, seed=None):
    assert mode in [
        'Train', 'Eval', 'Test', 'Gallery', 'Query', 'UnLabelTrain'
    ], "Dataset mode should be Train, Eval, Test, Gallery, Query, UnLabelTrain"
    assert mode in config.keys(), "{} config not in yaml".format(mode)
    # build dataset
    if use_dali:
        from ppcls.data.dataloader.dali import dali_dataloader
        return dali_dataloader(
            config,
            mode,
            paddle.device.get_device(),
            num_threads=config[mode]['loader']["num_workers"],
            seed=seed,
            enable_fuse=True)

    class_num = config.get("class_num", None)
    epochs = config.get("epochs", None)
    config_dataset = config[mode]['dataset']
    config_dataset = copy.deepcopy(config_dataset)
    dataset_name = config_dataset.pop('name')
    if 'batch_transform_ops' in config_dataset:
        batch_transform = config_dataset.pop('batch_transform_ops')
    else:
        batch_transform = None

    dataset = eval(dataset_name)(**config_dataset)

    logger.debug("build dataset({}) success...".format(dataset))

    # build sampler
    config_sampler = config[mode]['sampler']
    if config_sampler and "name" not in config_sampler:
        batch_sampler = None
        batch_size = config_sampler["batch_size"]
        drop_last = config_sampler["drop_last"]
        shuffle = config_sampler["shuffle"]
    else:
        sampler_name = config_sampler.pop("name")
        sampler_argspec = inspect.getargspec(eval(sampler_name).__init__).args
        if "total_epochs" in sampler_argspec:
            config_sampler.update({"total_epochs": epochs})
        batch_sampler = eval(sampler_name)(dataset, **config_sampler)

    logger.debug("build batch_sampler({}) success...".format(batch_sampler))

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
        batch_ops = create_operators(batch_transform, class_num)
        batch_collate_fn = mix_collate_fn
    else:
        batch_collate_fn = None

    # build dataloader
    config_loader = config[mode]['loader']
    num_workers = config_loader["num_workers"]
    use_shared_memory = config_loader["use_shared_memory"]

    init_fn = partial(
        worker_init_fn,
        num_workers=num_workers,
        rank=dist.get_rank(),
        seed=seed) if seed is not None else None

    if batch_sampler is None:
        data_loader = DataLoader(
            dataset=dataset,
            places=paddle.device.get_device(),
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=batch_collate_fn,
            worker_init_fn=init_fn)
    else:
        data_loader = DataLoader(
            dataset=dataset,
            places=paddle.device.get_device(),
            num_workers=num_workers,
            return_list=True,
            use_shared_memory=use_shared_memory,
            batch_sampler=batch_sampler,
            collate_fn=batch_collate_fn,
            worker_init_fn=init_fn)

    total_samples = len(
        data_loader.dataset) if not use_dali else data_loader.size
    max_iter = len(data_loader) - 1 if platform.system() == "Windows" else len(
        data_loader)
    data_loader.max_iter = max_iter
    data_loader.total_samples = total_samples

    logger.debug("build data_loader({}) success...".format(data_loader))
    return data_loader


# TODO(gaotingquan): perf
class DataIterator(object):
    def __init__(self, dataloader, use_dali=False):
        self.dataloader = dataloader
        self.use_dali = use_dali
        self.iterator = iter(dataloader)
        self.max_iter = dataloader.max_iter
        self.total_samples = dataloader.total_samples

    def get_batch(self):
        # fetch data batch from dataloader
        try:
            batch = next(self.iterator)
        except Exception:
            # NOTE: reset DALI dataloader manually
            if self.use_dali:
                self.dataloader.reset()
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        return batch


def build_dataloader(config, mode):
    if "class_num" in config["Global"]:
        global_class_num = config["Global"]["class_num"]
        if "class_num" not in config["Arch"]:
            config["Arch"]["class_num"] = global_class_num
            msg = f"The Global.class_num will be deprecated. Please use Arch.class_num instead. Arch.class_num has been set to {global_class_num}."
        else:
            msg = "The Global.class_num will be deprecated. Please use Arch.class_num instead. The Global.class_num has been ignored."
        logger.warning(msg)

    class_num = config["Arch"].get("class_num", None)
    config["DataLoader"].update({"class_num": class_num})
    config["DataLoader"].update({"epochs": config["Global"]["epochs"]})

    use_dali = config["Global"].get("use_dali", False)
    dataloader_dict = {
        "Train": None,
        "UnLabelTrain": None,
        "Eval": None,
        "Query": None,
        "Gallery": None,
        "GalleryQuery": None
    }
    if mode == 'train':
        train_dataloader = build(
            config["DataLoader"], "Train", use_dali, seed=None)

        if config["DataLoader"]["Train"].get("max_iter", None):
            # set max iteration per epoch mannualy, when training by iteration(s), such as XBM, FixMatch.
            max_iter = config["Train"].get("max_iter")
        update_freq = config["Global"].get("update_freq", 1)
        max_iter = train_dataloader.max_iter // update_freq * update_freq
        train_dataloader.max_iter = max_iter
        if config["DataLoader"]["Train"].get("convert_iterator", True):
            train_dataloader = DataIterator(train_dataloader, use_dali)
        dataloader_dict["Train"] = train_dataloader

    if config["DataLoader"].get('UnLabelTrain', None) is not None:
        dataloader_dict["UnLabelTrain"] = build(
            config["DataLoader"], "UnLabelTrain", use_dali, seed=None)

    if mode == "eval" or (mode == "train" and
                          config["Global"]["eval_during_train"]):
        if config["Global"]["eval_mode"] in ["classification", "adaface"]:
            dataloader_dict["Eval"] = build(
                config["DataLoader"], "Eval", use_dali, seed=None)
        elif config["Global"]["eval_mode"] == "retrieval":
            if len(config["DataLoader"]["Eval"].keys()) == 1:
                key = list(config["DataLoader"]["Eval"].keys())[0]
                dataloader_dict["GalleryQuery"] = build(
                    config["DataLoader"]["Eval"], key, use_dali)
            else:
                dataloader_dict["Gallery"] = build(
                    config["DataLoader"]["Eval"], "Gallery", use_dali)
                dataloader_dict["Query"] = build(config["DataLoader"]["Eval"],
                                                 "Query", use_dali)
    return dataloader_dict
