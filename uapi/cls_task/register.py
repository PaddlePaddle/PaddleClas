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

import os.path as osp

from ..base.register import register_suite_info, register_model_info
from .model import ClsModel
from .runner import ClsRunner
from .config import ClsConfig

# XXX: Hard-code relative path of repo root dir
_file_path = osp.realpath(__file__)
REPO_ROOT_PATH = osp.abspath(osp.join(osp.dirname(_file_path), '..', '..'))

register_suite_info({
    'suite_name': 'Cls',
    'model': ClsModel,
    'runner': ClsRunner,
    'config': ClsConfig,
    'runner_root_path': REPO_ROOT_PATH
})

PPLCNet_x1_0_CFG_PATH = osp.join(
    REPO_ROOT_PATH, 'ppcls/configs/ImageNet/PPLCNet/PPLCNet_x1_0.yaml')

register_model_info({
    'model_name': 'PPLCNet_x1_0',
    'suite': 'Cls',
    'config_path': PPLCNet_x1_0_CFG_PATH,
    'auto_compression_config_path': osp.join(
        REPO_ROOT_PATH, 'ppcls/configs/slim/PPLCNet_x1_0_quantization.yaml'),
    'supported_apis': ['train', 'predict', 'export', 'infer', 'compression']
})
