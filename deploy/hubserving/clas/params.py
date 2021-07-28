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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def get_default_confg():
    return {
        'Global': {
            "inference_model_dir": "../inference/",
            "batch_size": 1,
            'use_gpu': False,
            'use_fp16': False,
            'enable_mkldnn': False,
            'cpu_num_threads': 1,
            'use_tensorrt': False,
            'ir_optim': False,
            "gpu_mem": 8000,
            'enable_profile': False,
            "enable_benchmark": False
        },
        'PostProcess': {
            'main_indicator': 'Topk',
            'Topk': {
                'topk': 5,
                'class_id_map_file': './utils/imagenet1k_label_list.txt'
            }
        }
    }
