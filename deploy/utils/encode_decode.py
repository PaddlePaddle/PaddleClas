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

import base64

import numpy as np


def np_to_b64(images):
    img_str = base64.b64encode(images).decode('utf8')
    return img_str, images.shape


def b64_to_np(b64str, revert_params):
    shape = revert_params["shape"]
    dtype = revert_params["dtype"]
    dtype = getattr(np, dtype) if isinstance(str, type(dtype)) else dtype
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, dtype).reshape(shape)
    return data