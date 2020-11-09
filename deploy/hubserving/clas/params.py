# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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


class Config(object):
    pass


def read_params():
    cfg = Config()

    cfg.model_file = "./inference/cls_infer.pdmodel"
    cfg.params_file = "./inference/cls_infer.pdiparams"
    cfg.batch_size = 1
    cfg.use_gpu = False
    cfg.ir_optim = True
    cfg.gpu_mem = 8000
    cfg.use_fp16 = False
    cfg.use_tensorrt = False

    # params for preprocess
    cfg.resize_short = 256
    cfg.resize = 224
    cfg.normalize = True

    return cfg
