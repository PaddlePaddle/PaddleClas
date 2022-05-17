# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

__all__ == ["extract_subnet_weights"]

import os
import paddle


def extract_subnet_weights(distill_weights_path,
                           student_weights_path,
                           student_name="Student"):
    assert os.path.exists(distill_weights_path), \
        "Given distill_weights_path {} not exist.".format(distill_weights_path)
    # Load teacher and student weights 
    all_params = paddle.load(distill_weights_path)
    # Extract student weights
    student_prefix = student_name + "."
    s_params = {
        key[len(student_prefix):]: all_params[key]
        for key in all_params if student_prefix in key
    }
    assert len(
        s_params
    ) > 0, f"extracted params length must be > 0 but got {len(s_params)}"
    # Save subnet weights
    paddle.save(s_params, student_weights_path)
