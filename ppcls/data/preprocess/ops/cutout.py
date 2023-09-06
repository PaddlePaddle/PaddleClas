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

import numpy as np
import random

from PIL import Image, ImageDraw


class Cutout(object):
    def __init__(self, n_holes=1, length=112):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """ cutout_image """
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[y1:y2, x1:x2] = 0
        return img


class CutoutPIL(object):
    """
    Cutout use PIL backend.

    Args:
        n_holes (int): the number of cutout holes.
        cutout_factor (float): the ratio of cutout hole to image size.
        fill_color (tuple, optional): fill color, use random value if got None.
    """

    def __init__(self, n_holes=1, cutout_factor=0.5, fill_color=None):
        self.n_holes = n_holes
        self.cutout_factor = cutout_factor
        self.fill_color = fill_color

    def __call__(self, img):
        """ cutout_image """
        h, w = img.shape[:2]
        cutout_h = int(self.cutout_factor * h + 0.5)
        cutout_w = int(self.cutout_factor * w + 0.5)

        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - cutout_h // 2, 0, h).item()
            y2 = np.clip(y + cutout_h // 2, 0, h).item()
            x1 = np.clip(x - cutout_w // 2, 0, w).item()
            x2 = np.clip(x + cutout_w // 2, 0, w).item()

            if self.fill_color is None:
                target_color = (random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255))
            else:
                target_color = self.fill_color

            draw.rectangle((x1, y1, x2, y2), fill=target_color)

        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        return img
