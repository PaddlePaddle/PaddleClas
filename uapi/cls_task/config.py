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

from ..base import BaseConfig


class ClsConfig(BaseConfig):
    def update(self, list_like_obj):
        dict_ = override_config(self.dict, list_like_obj)
        self.reset_from_dict(dict_)
    
    def load(self, config_file_path):
        dict_ =  yaml.load(open(config_file_path, 'rb'), Loader=yaml.Loader)
        if not isinstance(dict_, dict):
            raise TypeError
        self.reset_from_dict(dict_)

    def dump(self, config_file_path):
        with open(config_file_path, 'w') as f:
            yaml.dump(self.dict, f, default_flow_style=False, sort_keys=False)

    def update_dataset(self, dataset_path, dataset_type=None):
        if dataset_type is None:
            dataset_type = 'ImageNetDataset'
        if dataset_type == 'ImageNetDataset':
            ds_cfg = [
                f'DataLoader.Train.dataset.name={dataset_type}',
                f'DataLoader.Train.dataset.image_root={dataset_path}',
                f'DataLoader.Train.dataset.cls_label_path={dataset_path}/train.txt',
                f'DataLoader.Eval.dataset.name={dataset_type}',
                f'DataLoader.Eval.dataset.image_root={dataset_path}',
                f'DataLoader.Eval.dataset.cls_label_path={dataset_path}/val.txt',
            ]
        else:
            raise ValueError(f"{dataset_type} is not supported.")
        self.update(ds_cfg)

    def update_batch_size(self, batch_size, mode='train'):
        _cfg = [
            f'DataLoader.Train.sampler.batch_size={batch_size}',
            f'DataLoader.Eval.sampler.batch_size={batch_size}',
        ]
        self.update(_cfg)

    def update_amp(self, amp):
        if amp is None:
            if 'AMP' in self.dict:
                self._dict.pop('AMP')
        else:
            _cfg = [
                'AMP.scale_loss=128',
                'AMP.use_dynamic_loss_scaling=True',
                f'AMP.level={amp}'
            ]
            self.update(_cfg)

    def update_device(self, device):
        device = device.split(':')[0]
        _cfg = [
            f'Global.device={device}'
        ]
        self.update(_cfg)
    
    def update_optimizer(self, optimizer_type):
        # Not yet implemented
        raise NotImplementedError

    def update_backbone(self, backbone_type):
        # Not yet implemented
        raise NotImplementedError

    def update_lr_scheduler(self, lr_scheduler_type):
        _cfg = [
            f'Optimizer.lr.learning_rate={lr_scheduler_type}',
            # 'Optimizer.lr.warmup_epoch': 0,
            # 'Optimizer.lr.name': 'Const',
        ]
        self.update(_cfg)

    def update_weight_decay(self, weight_decay):
        # Not yet implemented
        raise NotImplementedError