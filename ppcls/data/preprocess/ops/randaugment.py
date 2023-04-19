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

# This code is based on https://github.com/heartInsert/randaugment
# reference: https://arxiv.org/abs/1909.13719

import random
from paddle.vision.transforms import transforms as T

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    elif method == 'nearest':
        return Image.NEAREST
    raise NotImplementedError


def solarize_add(img, add, thresh=128, **__):
    lut = []
    for i in range(256):
        if i < thresh:
            lut.append(min(255, i + add))
        else:
            lut.append(i)
    if img.mode in ("L", "RGB"):
        if img.mode == "RGB" and len(lut) == 256:
            lut = lut + lut + lut
        return img.point(lut)
    else:
        return img


def cutout(image, pad_size, replace=0):
    image_np = np.array(image)
    image_height, image_width, _ = image_np.shape

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = np.random.randint(0, image_height + 1)
    cutout_center_width = np.random.randint(0, image_width + 1)

    lower_pad = np.maximum(0, cutout_center_height - pad_size)
    upper_pad = np.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = np.maximum(0, cutout_center_width - pad_size)
    right_pad = np.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [
        image_height - (lower_pad + upper_pad),
        image_width - (left_pad + right_pad)
    ]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = np.pad(np.zeros(
        cutout_shape, dtype=image_np.dtype),
                  padding_dims,
                  constant_values=1)
    mask = np.expand_dims(mask, -1)
    mask = np.tile(mask, [1, 1, 3])
    image_np = np.where(
        np.equal(mask, 0),
        np.full_like(
            image_np, fill_value=replace, dtype=image_np.dtype),
        image_np)
    return Image.fromarray(image_np)


class RandomApply(object):
    def __init__(self, p, transforms):
        self.p = p
        ts = []
        for t in transforms:
            for key in t.keys():
                ts.append(eval(key)(**t[key]))

        self.trans = T.Compose(ts)

    def __call__(self, img):
        if self.p < np.random.rand(1):
            return img
        timg = self.trans(img)
        return timg


class RandAugment(object):
    def __init__(self,
                 num_layers=2,
                 magnitude=5,
                 fillcolor=(128, 128, 128),
                 interpolation="bicubic"):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.fillcolor = tuple(fillcolor)
        self.interpolation = _pil_interp(interpolation)

        self.set_augmentation()
        self.set_func()

    def set_func(self):
        sig = lambda signed: random.choice([-1, 1]) if signed else 1

        self.func = {
            "shearX": lambda img, magnitude, signed: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * sig(signed), 0, 0, 1, 0),
                self.interpolation,
                fillcolor=self.fillcolor),
            "shearY": lambda img, magnitude, signed: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * sig(signed), 1, 0),
                self.interpolation,
                fillcolor=self.fillcolor),
            "translateX": lambda img, magnitude, signed: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * sig(signed), 0, 1, 0),
                self.interpolation,
                fillcolor=self.fillcolor),
            "translateY": lambda img, magnitude, signed: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * sig(signed)),
                self.interpolation,
                fillcolor=self.fillcolor),
            "rotate": lambda img, magnitude, signed: img.rotate(
                magnitude * sig(signed),
                self.interpolation,
                fillcolor=self.fillcolor),
            "color": lambda img, magnitude, signed: ImageEnhance.Color(img).enhance(
                1 + magnitude * sig(signed)),
            "posterize": lambda img, magnitude, signed:
                ImageOps.posterize(img, magnitude * sig(signed)),
            "solarize": lambda img, magnitude, signed:
                ImageOps.solarize(img, magnitude * sig(signed)),
            "contrast": lambda img, magnitude, signed:
                ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * sig(signed)),
            "sharpness": lambda img, magnitude, signed:
                ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * sig(signed)),
            "brightness": lambda img, magnitude, signed:
                ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * sig(signed)),
            "autocontrast": lambda img, *_:
                ImageOps.autocontrast(img),
            "equalize": lambda img, *_: ImageOps.equalize(img),
            "invert": lambda img, *_: ImageOps.invert(img),
        }

    def set_augmentation(self):
        abso_level = self.magnitude / 10
        self.level_map = {
            "shearX": (0.3 * abso_level, True),
            "shearY": (0.3 * abso_level, True),
            "translateX": (150.0 / 331.0 * abso_level, True),
            "translateY": (150.0 / 331.0 * abso_level, True),
            "rotate": (30 * abso_level, False),
            "color": (0.9 * abso_level, True),
            "posterize": (int(4.0 * abso_level), False),
            "solarize": (256.0 * abso_level, False),
            "contrast": (0.9 * abso_level, True),
            "sharpness": (0.9 * abso_level, True),
            "brightness": (0.9 * abso_level, True),
            "autocontrast": (0, False),
            "equalize": (0, False),
            "invert": (0, False),
        }

    def __call__(self, img):
        avaiable_op_names = list(self.level_map.keys())
        for _ in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, *self.level_map[op_name])
        return img


class RandAugmentV2(RandAugment):
    """Customed RandAugment for EfficientNetV2"""

    def set_func(self):
        super().set_func()
        sig = lambda signed: random.choice([-1, 1]) if signed else 1
        self.func.update({
            "solarize_add": lambda img, magnitude, signed: solarize_add(
                img, magnitude * sig(signed)),
            "cutout": lambda img, magnitude, signed: cutout(
                img, magnitude * sig(signed), replace=self.fillcolor[0]),
        })

    def set_augmentation(self):
        super().set_augmentation()
        abso_level = self.magnitude / 10
        self.level_map.update({
            "translateX": (100.0 * abso_level, True),
            "translateY": (100.0 * abso_level, True),
            "rotate": (30 * abso_level, True),
            "color": (1.8 * abso_level - 0.9, False),
            "solarize": (int(256.0 * abso_level), False),
            "solarize_add": (int(110.0 * abso_level), False),  # add
            "contrast": (1.8 * abso_level - 0.9, False),
            "sharpness": (1.8 * abso_level - 0.9, False),
            "brightness": (1.8 * abso_level - 0.9, False),
            "cutout": (int(40 * abso_level), False),  # add
        })


class RandAugmentV3(RandAugment):
    """Customed RandAugment for MobileViTv2"""

    def set_augmentation(self):
        super().set_augmentation()
        abso_level = self.magnitude / 10
        self.level_map.update({
            "rotate": (30 * abso_level, True),
            "posterize": (8 - int(4.0 * abso_level), False),
            "solarize": (255.0 * (1 - abso_level), False),
        })
