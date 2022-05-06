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

# This code is adapted from https://github.com/zhunzhong07/Random-Erasing, and refer to Timm(https://github.com/rwightman/pytorch-image-models).
# reference: https://arxiv.org/abs/1708.04896

from functools import partial

import math
import random

import numpy as np


class Pixels(object):
    def __init__(self, mode="const", mean=[0., 0., 0.]):
        self._mode = mode
        self._mean = np.array(mean)

    def __call__(self, h=224, w=224, c=3, channel_first=False):
        if self._mode == "rand":
            return np.random.normal(size=(
                1, 1, 3)) if not channel_first else np.random.normal(size=(
                    3, 1, 1))
        elif self._mode == "pixel":
            return np.random.normal(size=(
                h, w, c)) if not channel_first else np.random.normal(size=(
                    c, h, w))
        elif self._mode == "const":
            return np.reshape(self._mean, (
                1, 1, c)) if not channel_first else np.reshape(self._mean,
                                                               (c, 1, 1))
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(object):
    """RandomErasing.
    """

    def __init__(self,
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0., 0., 0.],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const'):
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (
            r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.get_pixels = Pixels(mode, mean)

    def __call__(self, img):
        if random.random() > self.EPSILON:
            return img

        for _ in range(self.attempt):
            if isinstance(img, np.ndarray):
                img_h, img_w, img_c = img.shape
                channel_first = False
            else:
                img_c, img_h, img_w = img.shape
                channel_first = True
            area = img_h * img_w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(*self.r1)
            if self.use_log_aspect:
                aspect_ratio = math.exp(aspect_ratio)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_w and h < img_h:
                pixels = self.get_pixels(h, w, img_c, channel_first)
                x1 = random.randint(0, img_h - h)
                y1 = random.randint(0, img_w - w)
                if img_c == 3:
                    if channel_first:
                        img[:, x1:x1 + h, y1:y1 + w] = pixels
                    else:
                        img[x1:x1 + h, y1:y1 + w, :] = pixels
                else:
                    if channel_first:
                        img[0, x1:x1 + h, y1:y1 + w] = pixels[0]
                    else:
                        img[x1:x1 + h, y1:y1 + w, 0] = pixels[:, :, 0]
                return img
        return img
