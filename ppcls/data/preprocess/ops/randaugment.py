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
from .operators import RawColorJitter
from paddle.vision.transforms import transforms as T

import numpy as np
from PIL import Image, ImageEnhance, ImageOps


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


class RandAugment(object):
    def __init__(self, num_layers=2, magnitude=5, fillcolor=(128, 128, 128)):
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_level = 10

        abso_level = self.magnitude / self.max_level
        self.level_map = {
            "shearX": 0.3 * abso_level,
            "shearY": 0.3 * abso_level,
            "translateX": 150.0 / 331 * abso_level,
            "translateY": 150.0 / 331 * abso_level,
            "rotate": 30 * abso_level,
            "color": 0.9 * abso_level,
            "posterize": int(4.0 * abso_level),
            "solarize": 256.0 * abso_level,
            "contrast": 0.9 * abso_level,
            "sharpness": 0.9 * abso_level,
            "brightness": 0.9 * abso_level,
            "autocontrast": 0,
            "equalize": 0,
            "invert": 0
        }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                   Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        rnd_ch_op = random.choice

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * rnd_ch_op([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * rnd_ch_op([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * rnd_ch_op([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "posterize": lambda img, magnitude:
                ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude:
                ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude:
                ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "sharpness": lambda img, magnitude:
                ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "brightness": lambda img, magnitude:
                ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "autocontrast": lambda img, _:
                ImageOps.autocontrast(img),
            "equalize": lambda img, _: ImageOps.equalize(img),
            "invert": lambda img, _: ImageOps.invert(img)
        }

    def __call__(self, img):
        avaiable_op_names = list(self.level_map.keys())
        for layer_num in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, self.level_map[op_name])
        return img


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


## RandAugment_EfficientNetV2 code below ##
class RandAugmentV2(RandAugment):
    """Customed RandAugment for EfficientNetV2"""

    def __init__(self,
                 num_layers=2,
                 magnitude=5,
                 progress_magnitude=None,
                 fillcolor=(128, 128, 128)):
        super().__init__(num_layers, magnitude, fillcolor)
        self.progress_magnitude = progress_magnitude
        abso_level = self.magnitude / self.max_level
        self.level_map = {
            "shearX": 0.3 * abso_level,
            "shearY": 0.3 * abso_level,
            "translateX": 100.0 * abso_level,
            "translateY": 100.0 * abso_level,
            "rotate": 30 * abso_level,
            "color": 1.8 * abso_level + 0.1,
            "posterize": int(4.0 * abso_level),
            "solarize": int(256.0 * abso_level),
            "solarize_add": int(110.0 * abso_level),
            "contrast": 1.8 * abso_level + 0.1,
            "sharpness": 1.8 * abso_level + 0.1,
            "brightness": 1.8 * abso_level + 0.1,
            "autocontrast": 0,
            "equalize": 0,
            "invert": 0,
            "cutout": int(40 * abso_level)
        }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                   Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        rnd_ch_op = random.choice

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * rnd_ch_op([-1, 1]), 0, 0, 1, 0),
                Image.NEAREST,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),
                Image.NEAREST,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * rnd_ch_op([-1, 1]), 0, 1, 0),
                Image.NEAREST,
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * rnd_ch_op([-1, 1])),
                Image.NEAREST,
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude * rnd_ch_op([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(magnitude),
            "posterize": lambda img, magnitude:
                ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude:
                ImageOps.solarize(img, magnitude),
            "solarize_add": lambda img, magnitude:
                solarize_add(img, magnitude),
            "contrast": lambda img, magnitude:
                ImageEnhance.Contrast(img).enhance(magnitude),
            "sharpness": lambda img, magnitude:
                ImageEnhance.Sharpness(img).enhance(magnitude),
            "brightness": lambda img, magnitude:
                ImageEnhance.Brightness(img).enhance(magnitude),
            "autocontrast": lambda img, _:
                ImageOps.autocontrast(img),
            "equalize": lambda img, _: ImageOps.equalize(img),
            "invert": lambda img, _: ImageOps.invert(img),
            "cutout": lambda img, magnitude: cutout(img, magnitude, replace=fillcolor[0])
        }
