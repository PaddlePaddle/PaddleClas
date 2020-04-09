# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import os
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as pff

trainers_num = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
trainer_id = int(os.environ.get("PADDLE_TRAINER_ID", 0))


def place():
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    return fluid.CUDAPlace(gpu_id)


def places():
    """
    Returns available running places, the numbers are usually
    indicated by 'export CUDA_VISIBLE_DEVICES= '
    Args:
    """

    if trainers_num <= 1:
        return pff.cuda_places()
    else:
        return place()
