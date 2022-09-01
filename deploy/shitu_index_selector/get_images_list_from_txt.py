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
import argparse
import base64
import numpy as np

import cv2
def get_image_list_from_txt(dataset, img_file, img_path=None):
    imgs_lists = {}
    if img_file is None or not os.path.exists(img_file):
        print("infer_imgs file of {} not found".format(dataset))
        return None

    img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
    
    f = open(img_file)
    file_list = f.readlines()
    for img in file_list:
        label = img.split()[1]
        if label not in imgs_lists:
            imgs_lists[label] = []
        img = img.split()[0]
        if img.split('.')[-1] in img_end:
            if img_path is not None:
                img = os.path.join(img_path, img)
            if img is None or not os.path.exists(img):
                return None
            imgs_lists[label].append(img)

    if len(imgs_lists) == 0:
        return None
    # imgs_lists = sorted(imgs_lists)

    return imgs_lists

