# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os

import yaml
import collections

from ppcls.utils.config import get_config, override_config

from ..cls_task import ClsConfig


class ShiTuConfig(ClsConfig):

    def _update_dataset_config(self, dataset_root_path):
        _cfg = [
            'DataLoader.Train.dataset.name=ImageNetDataset',
            f'DataLoader.Train.dataset.image_root={dataset_root_path}',
            f'DataLoader.Train.dataset.cls_label_path={dataset_root_path}/train.txt',
            'DataLoader.Eval.Query.dataset.name=VeriWild',
            f'DataLoader.Eval.Query.dataset.image_root={dataset_root_path}',
            f'DataLoader.Eval.Query.dataset.cls_label_path={dataset_root_path}/val.txt',
            'DataLoader.Eval.Gallery.dataset.name=VeriWild',
            f'DataLoader.Eval.Gallery.dataset.image_root={dataset_root_path}',
            f'DataLoader.Eval.Gallery.dataset.cls_label_path={dataset_root_path}/val.txt',
        ]
        self.update(_cfg)

    def _update_batch_size_config(self, batch_size):
        _cfg = [
            f'DataLoader.Train.sampler.batch_size={batch_size}',
            f'DataLoader.Eval.Query.sampler.batch_size={batch_size}',
            f'DataLoader.Eval.Gallery.sampler.batch_size={batch_size}',
        ]
        self.update(_cfg)