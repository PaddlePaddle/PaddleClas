# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
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
import copy
import shutil
from functools import partial
import importlib
import numpy as np
import paddle
import paddle.nn.functional as F


def build_postprocess(config):
    if config is None:
        return None

    mod = importlib.import_module(__name__)
    config = copy.deepcopy(config)

    main_indicator = config.pop(
        "main_indicator") if "main_indicator" in config else None
    main_indicator = main_indicator if main_indicator else ""

    func_list = []
    for func in config:
        func_list.append(getattr(mod, func)(**config[func]))
    return PostProcesser(func_list, main_indicator)


class PostProcesser(object):
    def __init__(self, func_list, main_indicator="Topk"):
        self.func_list = func_list
        self.main_indicator = main_indicator

    def __call__(self, x, image_file=None):
        rtn = None
        for func in self.func_list:
            tmp = func(x, image_file)
            if type(func).__name__ in self.main_indicator:
                rtn = tmp
        return rtn


class ThreshOutput(object):
    def __init__(self, threshold, label_0="0", label_1="1"):
        self.threshold = threshold
        self.label_0 = label_0
        self.label_1 = label_1

    def __call__(self, x, file_names=None):
        y = []
        for idx, probs in enumerate(x):
            score = probs[1]
            if score < self.threshold:
                result = {
                    "class_ids": [0],
                    "scores": [1 - score],
                    "label_names": [self.label_0]
                }
            else:
                result = {
                    "class_ids": [1],
                    "scores": [score],
                    "label_names": [self.label_1]
                }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            y.append(result)
        return y


class Topk(object):
    def __init__(self, topk=1, class_id_map_file=None):
        assert isinstance(topk, (int, ))
        self.class_id_map = self.parse_class_id_map(class_id_map_file)
        self.topk = topk

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
                    partition = line.split("\n")[0].partition(" ")
                    class_id_map[int(partition[0])] = str(partition[-1])
        except Exception as ex:
            print(ex)
            class_id_map = None
        return class_id_map

    def __call__(self, x, file_names=None, multilabel=False):
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        y = []
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-self.topk:][::-1].astype(
                "int32") if not multilabel else np.where(
                    probs >= 0.5)[0].astype("int32")
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
            }
            if file_names is not None:
                result["file_name"] = file_names[idx]
            if label_name_list is not None:
                result["label_names"] = label_name_list
            y.append(result)
        return y


class MultiLabelTopk(Topk):
    def __init__(self, topk=1, class_id_map_file=None):
        super().__init__()

    def __call__(self, x, file_names=None):
        return super().__call__(x, file_names, multilabel=True)


class SavePreLabel(object):
    def __init__(self, save_dir):
        if save_dir is None:
            raise Exception(
                "Please specify save_dir if SavePreLabel specified.")
        self.save_dir = partial(os.path.join, save_dir)

    def __call__(self, x, file_names=None):
        if file_names is None:
            return
        assert x.shape[0] == len(file_names)
        for idx, probs in enumerate(x):
            index = probs.argsort(axis=0)[-1].astype("int32")
            self.save(index, file_names[idx])

    def save(self, id, image_file):
        output_dir = self.save_dir(str(id))
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(image_file, output_dir)


class Binarize(object):
    def __init__(self, method="round"):
        self.method = method
        self.unit = np.array([[128, 64, 32, 16, 8, 4, 2, 1]]).T

    def __call__(self, x, file_names=None):
        if self.method == "round":
            x = np.round(x + 1).astype("uint8") - 1

        if self.method == "sign":
            x = ((np.sign(x) + 1) / 2).astype("uint8")

        embedding_size = x.shape[1]
        assert embedding_size % 8 == 0, "The Binary index only support vectors with sizes multiple of 8"

        byte = np.zeros([x.shape[0], embedding_size // 8], dtype=np.uint8)
        for i in range(embedding_size // 8):
            byte[:, i:i + 1] = np.dot(x[:, i * 8:(i + 1) * 8], self.unit)

        return byte


class PersonAttribute(object):
    def __init__(self,
                 threshold=0.5,
                 glasses_threshold=0.3,
                 hold_threshold=0.6):
        self.threshold = threshold
        self.glasses_threshold = glasses_threshold
        self.hold_threshold = hold_threshold

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        age_list = ['AgeLess18', 'Age18-60', 'AgeOver60']
        direct_list = ['Front', 'Side', 'Back']
        bag_list = ['HandBag', 'ShoulderBag', 'Backpack']
        upper_list = ['UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice']
        lower_list = [
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts',
            'Skirt&Dress'
        ]
        batch_res = []
        for res in batch_preds:
            res = res.tolist()
            label_res = []
            # gender 
            gender = 'Female' if res[22] > self.threshold else 'Male'
            label_res.append(gender)
            # age
            age = age_list[np.argmax(res[19:22])]
            label_res.append(age)
            # direction 
            direction = direct_list[np.argmax(res[23:])]
            label_res.append(direction)
            # glasses
            glasses = 'Glasses: '
            if res[1] > self.glasses_threshold:
                glasses += 'True'
            else:
                glasses += 'False'
            label_res.append(glasses)
            # hat
            hat = 'Hat: '
            if res[0] > self.threshold:
                hat += 'True'
            else:
                hat += 'False'
            label_res.append(hat)
            # hold obj
            hold_obj = 'HoldObjectsInFront: '
            if res[18] > self.hold_threshold:
                hold_obj += 'True'
            else:
                hold_obj += 'False'
            label_res.append(hold_obj)
            # bag
            bag = bag_list[np.argmax(res[15:18])]
            bag_score = res[15 + np.argmax(res[15:18])]
            bag_label = bag if bag_score > self.threshold else 'No bag'
            label_res.append(bag_label)
            # upper
            upper_res = res[4:8]
            upper_label = 'Upper:'
            sleeve = 'LongSleeve' if res[3] > res[2] else 'ShortSleeve'
            upper_label += ' {}'.format(sleeve)
            for i, r in enumerate(upper_res):
                if r > self.threshold:
                    upper_label += ' {}'.format(upper_list[i])
            label_res.append(upper_label)
            # lower
            lower_res = res[8:14]
            lower_label = 'Lower: '
            has_lower = False
            for i, l in enumerate(lower_res):
                if l > self.threshold:
                    lower_label += ' {}'.format(lower_list[i])
                    has_lower = True
            if not has_lower:
                lower_label += ' {}'.format(lower_list[np.argmax(lower_res)])

            label_res.append(lower_label)
            # shoe
            shoe = 'Boots' if res[14] > self.threshold else 'No boots'
            label_res.append(shoe)

            threshold_list = [0.5] * len(res)
            threshold_list[1] = self.glasses_threshold
            threshold_list[18] = self.hold_threshold
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res


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

    def __call__(self, batch_preds, file_names=None):
        # postprocess output of predictor
        batch_res = []
        for res in batch_preds:
            res = res.tolist()
            label_res = []
            color_idx = np.argmax(res[:10])
            type_idx = np.argmax(res[10:])
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
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res
