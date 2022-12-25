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
from __future__ import absolute_import, division, print_function

import time

import numpy as np

from ppcls.data import build_dataloader
from ppcls.utils import logger

from .train import train_epoch


def train_epoch_efficientnetv2(engine, epoch_id, print_batch_step):
    # 1. Build training hyper-parameters for different training stage
    num_stage = 4
    ratio_list = [(i + 1) / num_stage for i in range(num_stage)]
    ram_list = np.linspace(5, 10, num_stage)
    # dropout_rate_list = np.linspace(0.0, 0.2, num_stage)
    stones = [
        int(engine.config["Global"]["epochs"] * ratio_list[i])
        for i in range(num_stage)
    ]
    image_size_list = [
        int(128 + (300 - 128) * ratio_list[i]) for i in range(num_stage)
    ]
    stage_id = 0
    for i in range(num_stage):
        if epoch_id > stones[i]:
            stage_id = i + 1

    # 2. Adjust training hyper-parameters for different training stage
    if not hasattr(engine, 'last_stage') or engine.last_stage < stage_id:
        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][1][
            "RandCropImage"]["size"] = image_size_list[stage_id]
        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][3][
            "RandAugment"]["magnitude"] = ram_list[stage_id]
        engine.train_dataloader = build_dataloader(
            engine.config["DataLoader"],
            "Train",
            engine.device,
            engine.use_dali,
            seed=epoch_id)
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.last_stage = stage_id
    logger.info(
        f"Training stage: [{stage_id+1}/{num_stage}](random_aug_magnitude={ram_list[stage_id]}, train_image_size={image_size_list[stage_id]})"
    )

    # 3. Train one epoch as usual at current stage
    train_epoch(engine, epoch_id, print_batch_step)
