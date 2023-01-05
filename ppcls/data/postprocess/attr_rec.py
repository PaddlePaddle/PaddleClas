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


class PersonAttribute(object):
    def __init__(self,
                 threshold=0.5,
                 glasses_threshold=0.3,
                 hold_threshold=0.6):
        self.threshold = threshold
        self.glasses_threshold = glasses_threshold
        self.hold_threshold = hold_threshold

    def __call__(self, x, file_names=None):
        if isinstance(x, dict):
            x = x['logits']
        assert isinstance(x, paddle.Tensor)
        if file_names is not None:
            assert x.shape[0] == len(file_names)
        x = F.sigmoid(x).numpy()

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
        for idx, res in enumerate(x):
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


class FaceAttribute(object):
    def __init__(self, threshold=0.65, convert_cn=False):
        self.threshold = threshold
        self.convert_cn = convert_cn

    def __call__(self, x, file_names=None):
        if isinstance(x, dict):
            x = x['logits']
        assert isinstance(x, paddle.Tensor)

        if file_names is not None:
            assert x.shape[0] == len(file_names)
        x = F.sigmoid(x).numpy()

        attribute_list = [
            ["CheekWhiskers", "刚长出的双颊胡须"], ["ArchedEyebrows", "柳叶眉"],
            ["Attractive", "吸引人的"], ["BagsUnderEyes", "眼袋"], ["Bald", "秃头"],
            ["Bangs", "刘海"], ["BigLips", "大嘴唇"], ["BigNose", "大鼻子"],
            ["BlackHair", "黑发"], ["BlondHair", "金发"], ["Blurry", "模糊的"],
            ["BrownHair", "棕发"], ["BushyEyebrows", "浓眉"], ["Chubby", "圆胖的"],
            ["DoubleChin", "双下巴"], ["Eyeglasses", "带眼镜"], ["Goatee", "山羊胡子"],
            ["GrayHair", "灰发或白发"], ["HeavyMakeup", "浓妆"],
            ["HighCheekbones", "高颧骨"], ["Male", "男性"],
            ["MouthSlightlyOpen", "微微张开嘴巴"], ["Mustache", "胡子"],
            ["NarrowEyes", "细长的眼睛"], ["NoBeard", "无胡子"],
            ["OvalFace", "椭圆形的脸"], ["PaleSkin", "苍白的皮肤"],
            ["PointyNose", "尖鼻子"], ["RecedingHairline", "发际线后移"],
            ["RosyCheeks", "红润的双颊"], ["Sideburns", "连鬓胡子"], ["Smiling", "微笑"],
            ["StraightHair", "直发"], ["WavyHair", "卷发"],
            ["WearingEarrings", "戴着耳环"], ["WearingHat", "戴着帽子"],
            ["WearingLipstick", "涂了唇膏"], ["WearingNecklace", "戴着项链"],
            ["WearingNecktie", "戴着领带"], ["Young", "年轻人"]
        ]
        gender_list = [["Male", "男性"], ["Female", "女性"]]
        age_list = [["Young", "年轻人"], ["Old", "老年人"]]
        batch_res = []
        index = 1 if self.convert_cn else 0
        for idx, res in enumerate(x):
            res = res.tolist()
            label_res = []
            threshold_list = [self.threshold] * len(res)
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            for i, value in enumerate(pred_res):
                if i == 20:
                    label_res.append(gender_list[0][index]
                                     if value == 1 else gender_list[1][index])
                elif i == 39:
                    label_res.append(age_list[0][index]
                                     if value == 1 else age_list[1][index])
                else:
                    if value == 1:
                        label_res.append(attribute_list[i][index])
            batch_res.append({"attributes": label_res, "output": pred_res})
        return batch_res


class TableAttribute(object):
    def __init__(
            self,
            source_threshold=0.5,
            number_threshold=0.5,
            color_threshold=0.5,
            clarity_threshold=0.5,
            obstruction_threshold=0.5,
            angle_threshold=0.5, ):
        self.source_threshold = source_threshold
        self.number_threshold = number_threshold
        self.color_threshold = color_threshold
        self.clarity_threshold = clarity_threshold
        self.obstruction_threshold = obstruction_threshold
        self.angle_threshold = angle_threshold

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
            source = 'Scanned' if res[0] > self.source_threshold else 'Photo'
            number = 'Little' if res[1] > self.number_threshold else 'Numerous'
            color = 'Black-and-White' if res[
                2] > self.color_threshold else 'Multicolor'
            clarity = 'Clear' if res[3] > self.clarity_threshold else 'Blurry'
            obstruction = 'Without-Obstacles' if res[
                4] > self.number_threshold else 'With-Obstacles'
            angle = 'Horizontal' if res[
                5] > self.number_threshold else 'Tilted'

            label_res = [source, number, color, clarity, obstruction, angle]

            threshold_list = [
                self.source_threshold, self.number_threshold,
                self.color_threshold, self.clarity_threshold,
                self.obstruction_threshold, self.angle_threshold
            ]
            pred_res = (np.array(res) > np.array(threshold_list)
                        ).astype(np.int8).tolist()
            batch_res.append({
                "attributes": label_res,
                "output": pred_res,
                "file_name": file_names[idx]
            })
        return batch_res
