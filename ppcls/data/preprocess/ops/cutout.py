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

# This code is based on https://github.com/uoguelph-mlrg/Cutout
# reference: https://arxiv.org/abs/1708.04552

import random

import numpy as np
import cv2


class Cutout(object):
    def __init__(self, n_holes=1, length=112, fill_value=(0, 0, 0)):
        self.n_holes = n_holes
        self.length = length
        if fill_value == 'none' or fill_value is None:
            self.fill_value = None

    def __call__(self, img):
        """ cutout_image """
        h, w = img.shape[:2]

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            fill_value = self.fill_value
            if fill_value is None:
                if img.ndim == 2:
                    fill_value = random.randint(0, 255)
                else:
                    fill_value = [random.randint(0, 255),
                                  random.randint(0, 255),
                                  random.randint(0, 255)]

                img = cv2.rectangle(img, (x1, y1), (x2, y2), fill_value, -1)

        return img
