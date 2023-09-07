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

            if img.ndim == 2:
                if self.fill_value is None:
                    fill_value = random.randint(0, 255)
                else:
                    fill_value = self.fill_value
                img[y1:y2, x1:x2] = fill_value
            else:
                if self.fill_value is None:
                    fill_value = [random.randint(0, 255),
                                  random.randint(0, 255),
                                  random.randint(0, 255)]
                else:
                    fill_value = self.fill_value
                img[y1:y2, x1:x2, 0] = fill_value[0]
                img[y1:y2, x1:x2, 1] = fill_value[1]
                img[y1:y2, x1:x2, 2] = fill_value[2]

        return img
