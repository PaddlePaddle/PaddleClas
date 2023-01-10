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

from ppcls.data import build_dataloader
from ppcls.engine.train.utils import type_name
from ppcls.utils import logger

from .train import train_epoch


def train_epoch_progressive(engine, epoch_id, print_batch_step):
    # 1. Build training hyper-parameters for different training stage
    num_stage = 4
    ratio_list = [(i + 1) / num_stage for i in range(num_stage)]
    stones = [
        int(engine.config["Global"]["epochs"] * ratio_list[i])
        for i in range(num_stage)
    ]
    stage_id = 0
    for i in range(num_stage):
        if epoch_id > stones[i]:
            stage_id = i + 1

    # 2. Adjust training hyper-parameters for different training stage
    if not hasattr(engine, 'last_stage') or engine.last_stage < stage_id:
        cur_dropout_rate = 0.0

        def _change_dp_func(m):
            global cur_dropout_rate
            if type_name(m) == "Head" and hasattr(m, "_dropout"):
                m._dropout.p = m.dropout_rate[stage_id]
                cur_dropout_rate = m.dropout_rate[stage_id]

        engine.model.apply(_change_dp_func)

        cur_image_size = engine.config["DataLoader"]["Train"]["dataset"][
            "transform_ops"][1]["RandCropImage"]["progress_size"][stage_id]
        cur_magnitude = engine.config["DataLoader"]["Train"]["dataset"][
            "transform_ops"][3]["RandAugmentV2"]["progress_magnitude"][
                stage_id]
        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][1][
            "RandCropImage"]["size"] = cur_image_size
        engine.config["DataLoader"]["Train"]["dataset"]["transform_ops"][3][
            "RandAugmentV2"]["magnitude"] = cur_magnitude
        engine.train_dataloader = build_dataloader(
            engine.config["DataLoader"],
            "Train",
            engine.device,
            engine.use_dali,
            seed=epoch_id)
        engine.train_dataloader_iter = iter(engine.train_dataloader)
        engine.last_stage = stage_id
    logger.info(f"Training stage: [{stage_id+1}/{num_stage}]("
                f"random_aug_magnitude={cur_magnitude}, "
                f"train_image_size={cur_image_size}, "
                f"dropout_rate={cur_dropout_rate}"
                f")")

    # 3. Train one epoch as usual at current stage
    train_epoch(engine, epoch_id, print_batch_step)
