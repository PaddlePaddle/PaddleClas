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
import paddle
import numpy as np

from .operators import format_data

def _get_pixels(per_pixel, rand_color, patch_size, dtype=paddle.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        # return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        return paddle.normal(shape=patch_size)
    elif rand_color:
        # return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        return paddle.normal(shape=[patch_size[0], 1, 1])
    else:
        # return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)
        return paddle.zeros([patch_size[0], 1, 1], dtype=dtype)

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

    @format_data
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

class BeitV2RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf

        This variant of BeitV2RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """

    def __init__(
            self,
            probability=0.5, min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
            mode='const', min_count=1, max_count=None, num_splits=0, device='cuda'):
        self.probability = probability
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_count = min_count
        self.max_count = max_count or min_count
        self.num_splits = num_splits
        mode = mode.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'
        self.device = device

    def _erase(self, img, chan, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, h, w),
                        dtype=dtype, device=self.device)
                    break

    def __call__(self, input):
        if len(input.shape) == 3:
            self._erase(input, *input.shape, input.dtype)
        else:
            batch_size, chan, img_h, img_w = input.shape
            # skip first slice of batch if num_splits is set (for clean portion of samples)
            batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
            for i in range(batch_start, batch_size):
                self._erase(input[i], chan, img_h, img_w, input.dtype)
        return input