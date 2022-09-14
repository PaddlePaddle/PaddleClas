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

import os
import numpy as np

import cv2
from collections import defaultdict


def get_image_list_from_txt(dataset, img_file, dir_path=None):
    imgs_lists = {}
    if img_file is None or not os.path.exists(img_file):
        print("infer_imgs file of {} not found".format(dataset))
        return None

    imgs_lists = defaultdict(list)
    img_end = ['jpg', 'png', 'jpeg', 'bmp']

    file_list = open(img_file).readlines()
    for img in file_list:
        [img, label] = img.split()
        if img.split('.')[-1].lower() in img_end:
            if dir_path is not None:
                img = os.path.join(dir_path, img)
            if img is None or not os.path.exists(img):
                return None
            imgs_lists[label].append(img)

    if len(imgs_lists) == 0:
        return None

    return imgs_lists
