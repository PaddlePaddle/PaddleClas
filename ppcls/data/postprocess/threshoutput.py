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


class MultiLabelThreshOutput(object):
    def __init__(self, threshold=0.5, class_id_map_file=None, delimiter=None):
        self.threshold = threshold
        self.delimiter = delimiter if delimiter is not None else " "
        self.class_id_map = self.parse_class_id_map(class_id_map_file)

    def parse_class_id_map(self, class_id_map_file):
        if class_id_map_file is None:
            return None
        if not os.path.exists(class_id_map_file):
            print(
                "Warning: If want to use your own label_dict, please input legal path!\nOtherwise label_names will be empty!"
            )
            return None

        try:
            class_id_map = {}
            with open(class_id_map_file, "r") as fin:
                lines = fin.readlines()
                for line in lines:
                    partition = line.split("\n")[0].partition(self.delimiter)
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            print(ex)
            class_id_map = None
        return class_id_map

    def __call__(self, x, file_names=None):
        y = []
        x = F.sigmoid(x).numpy()
        for idx, probs in enumerate(x):
            index = np.where(probs >= self.threshold)[0].astype("int32")
            clas_id_list = []
            score_list = []
            label_name_list = []
            for i in index:
                clas_id_list.append(i.item())
                score_list.append(probs[i].item())
                if self.class_id_map is not None:
                    label_name_list.append(self.class_id_map[i.item()])
            result = {
                "class_ids": clas_id_list,
                "scores": np.around(
                    score_list, decimals=5).tolist(),
                "label_names": label_name_list    
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y
