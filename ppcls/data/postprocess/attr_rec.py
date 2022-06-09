# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import numpy as np
import paddle
import paddle.nn.functional as F


class VehicleAttribute(object):
    def __init__(self, color_threshold=0.5, type_threshold=0.5):
        self.color_threshold = color_threshold
        self.type_threshold = type_threshold
        self.color_list = [
            "yellow", "orange", "green", "gray", "red", "blue", "white",
            "golden", "brown", "black"
        ]
        self.type_list = [
            "sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus",
            "truck", "estate"
        ]

    def __call__(self, x, file_names=None):
        if isinstance(x, dict):
            x = x['logits']
        assert isinstance(x, paddle.Tensor)
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        x = F.sigmoid(x).numpy()

        # postprocess output of predictor
        batch_res = []
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])
            print(color_idx, type_idx)
            if res[color_idx] >= self.color_threshold:
                color_info = f"Color: ({self.color_list[color_idx]}, prob: {res[color_idx]})"
            else:
                color_info = "Color unknown"

            if res[type_idx + 10] >= self.type_threshold:
                type_info = f"Type: ({self.type_list[type_idx]}, prob: {res[type_idx + 10]})"
            else:
                type_info = "Type unknown"

            label_res = f"{color_info}, {type_info}"

            threshold_list = [self.color_threshold
                              ] * 10 + [self.type_threshold] * 9
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({
                "attr": label_res,
                "pred": pred_res,
                "file_name": file_names[idx]
            })
        return batch_res
