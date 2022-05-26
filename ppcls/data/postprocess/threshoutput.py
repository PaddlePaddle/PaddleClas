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

import paddle.nn.functional as F


class ThreshOutput(object):
    def __init__(self, threshold, label_0="0", label_1="1"):
        self.threshold = threshold
        self.label_0 = label_0
        self.label_1 = label_1

    def __call__(self, x, file_names=None):
        y = []
        x = F.softmax(x, axis=-1).numpy()
        for idx, probs in enumerate(x):
            score = probs[1]
            if score < self.threshold:
                result = {"class_ids": [0], "scores":  [1 - score], "label_names": [self.label_0]}
            else:
                result = {"class_ids": [1], "scores": [score], "label_names": [self.label_1]}
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y
