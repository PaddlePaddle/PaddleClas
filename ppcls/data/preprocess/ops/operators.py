# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from functools import partial
import six
import math
import random
import cv2
import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION
from paddle.vision.transforms import ColorJitter as RawColorJitter
from paddle.vision.transforms import ToTensor, Normalize, RandomHorizontalFlip, RandomResizedCrop
from paddle.vision.transforms import functional as F
from .autoaugment import ImageNetPolicy
from .functional import augmentations
from ppcls.utils import logger


class UnifiedResize(object):
    def __init__(self, interpolation=None, backend="cv2", return_numpy=True):
        _cv2_interp_from_str = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'area': cv2.INTER_AREA,
            'bicubic': cv2.INTER_CUBIC,
            'lanczos': cv2.INTER_LANCZOS4,
            'random': (cv2.INTER_LINEAR, cv2.INTER_CUBIC)
        }
        _pil_interp_from_str = {
            'nearest': Image.NEAREST,
            'bilinear': Image.BILINEAR,
            'bicubic': Image.BICUBIC,
            'box': Image.BOX,
            'lanczos': Image.LANCZOS,
            'hamming': Image.HAMMING,
            'random': (Image.BILINEAR, Image.BICUBIC)
        }

        def _cv2_resize(src, size, resample):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            return cv2.resize(src, size, interpolation=resample)

        def _pil_resize(src, size, resample, return_numpy=True):
            if isinstance(resample, tuple):
                resample = random.choice(resample)
            if isinstance(src, np.ndarray):
                pil_img = Image.fromarray(src)
            else:
                pil_img = src
            pil_img = pil_img.resize(size, resample)
            if return_numpy:
                return np.asarray(pil_img)
            return pil_img

        if backend.lower() == "cv2":
            if isinstance(interpolation, str):
                interpolation = _cv2_interp_from_str[interpolation.lower()]
            # compatible with opencv < version 4.4.0
            elif interpolation is None:
                interpolation = cv2.INTER_LINEAR
            self.resize_func = partial(_cv2_resize, resample=interpolation)
        elif backend.lower() == "pil":
            if isinstance(interpolation, str):
                interpolation = _pil_interp_from_str[interpolation.lower()]
            self.resize_func = partial(
                _pil_resize, resample=interpolation, return_numpy=return_numpy)
        else:
            logger.warning(
                f"The backend of Resize only support \"cv2\" or \"PIL\". \"f{backend}\" is unavailable. Use \"cv2\" instead."
            )
            self.resize_func = cv2.resize

    def __call__(self, src, size):
        if isinstance(size, list):
            size = tuple(size)
        return self.resize_func(src, size)


class RandomInterpolationAugment(object):
    def __init__(self, prob):
        self.prob = prob

    def _aug(self, img):
        img_shape = img.shape
        side_ratio = np.random.uniform(0.2, 1.0)
        small_side = int(side_ratio * img_shape[0])
        interpolation = np.random.choice([
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        ])
        small_img = cv2.resize(
            img, (small_side, small_side), interpolation=interpolation)
        interpolation = np.random.choice([
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA,
            cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        ])
        aug_img = cv2.resize(
            small_img, (img_shape[1], img_shape[0]),
            interpolation=interpolation)
        return aug_img

    def __call__(self, img):
        if np.random.random() < self.prob:
            if isinstance(img, np.ndarray):
                return self._aug(img)
            else:
                pil_img = np.array(img)
                aug_img = self._aug(pil_img)
                img = Image.fromarray(aug_img.astype(np.uint8))
                return img
        else:
            return img


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


class DecodeImage(object):
    """ decode image """

    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.to_rgb = to_rgb
        self.to_np = to_np  # to numpy
        self.channel_first = channel_first  # only enabled when to_np is True

    def __call__(self, img):
        if not isinstance(img, np.ndarray):
            if six.PY2:
                assert type(img) is str and len(
                    img) > 0, "invalid input 'img' in DecodeImage"
            else:
                assert type(img) is bytes and len(
                    img) > 0, "invalid input 'img' in DecodeImage"
            data = np.frombuffer(img, dtype='uint8')
            img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


class ResizeImage(object):
    """ resize image """

    def __init__(self,
                 size=None,
                 resize_short=None,
                 interpolation=None,
                 backend="cv2",
                 return_numpy=True):
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

        self._resize_func = UnifiedResize(
            interpolation=interpolation,
            backend=backend,
            return_numpy=return_numpy)

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[:2]
        else:
            img_w, img_h = img.size

        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        return self._resize_func(img, (w, h))


class CropWithPadding(RandomResizedCrop):
    """
    crop image and padding to original size
    """

    def __init__(self,
                 prob=1,
                 padding_num=0,
                 size=224,
                 scale=(0.08, 1.0),
                 ratio=(3. / 4, 4. / 3),
                 interpolation='bilinear',
                 key=None):
        super().__init__(size, scale, ratio, interpolation, key)
        self.prob = prob
        self.padding_num = padding_num

    def __call__(self, img):
        is_cv2_img = False
        if isinstance(img, np.ndarray):
            flag = True
        if np.random.random() < self.prob:
            # RandomResizedCrop augmentation
            new = np.zeros_like(np.array(img)) + self.padding_num
            #  orig_W, orig_H = F._get_image_size(sample)
            orig_W, orig_H = self._get_image_size(img)
            i, j, h, w = self._get_param(img)
            cropped = F.crop(img, i, j, h, w)
            new[i:i + h, j:j + w, :] = np.array(cropped)
            if not isinstance:
                new = Image.fromarray(new.astype(np.uint8))
            return new
        else:
            return img

    def _get_image_size(self, img):
        if F._is_pil_image(img):
            return img.size
        elif F._is_numpy_image(img):
            return img.shape[:2][::-1]
        elif F._is_tensor_image(img):
            return img.shape[1:][::-1]  # chw
        else:
            raise TypeError("Unexpected type {}".format(type(img)))


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class Padv2(object):
    def __init__(self,
                 size=None,
                 size_divisor=32,
                 pad_mode=0,
                 offsets=None,
                 fill_value=(127.5, 127.5, 127.5)):
        """
        Pad image to a specified size or multiple of size_divisor.
        Args:
            size (int, list): image target size, if None, pad to multiple of size_divisor, default None
            size_divisor (int): size divisor, default 32
            pad_mode (int): pad mode, currently only supports four modes [-1, 0, 1, 2]. if -1, use specified offsets
                if 0, only pad to right and bottom. if 1, pad according to center. if 2, only pad left and top
            offsets (list): [offset_x, offset_y], specify offset while padding, only supported pad_mode=-1
            fill_value (bool): rgb value of pad area, default (127.5, 127.5, 127.5)
        """

        if not isinstance(size, (int, list)):
            raise TypeError(
                "Type of target_size is invalid when random_size is True. \
                            Must be List, now is {}".format(type(size)))

        if isinstance(size, int):
            size = [size, size]

        assert pad_mode in [
            -1, 0, 1, 2
        ], 'currently only supports four modes [-1, 0, 1, 2]'
        if pad_mode == -1:
            assert offsets, 'if pad_mode is -1, offsets should not be None'

        self.size = size
        self.size_divisor = size_divisor
        self.pad_mode = pad_mode
        self.fill_value = fill_value
        self.offsets = offsets

    def apply_image(self, image, offsets, im_size, size):
        x, y = offsets
        im_h, im_w = im_size
        h, w = size
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array(self.fill_value, dtype=np.float32)
        canvas[y:y + im_h, x:x + im_w, :] = image.astype(np.float32)
        return canvas

    def __call__(self, img):
        im_h, im_w = img.shape[:2]
        if self.size:
            w, h = self.size
            assert (
                im_h <= h and im_w <= w
            ), '(h, w) of target size should be greater than (im_h, im_w)'
        else:
            h = int(np.ceil(im_h / self.size_divisor) * self.size_divisor)
            w = int(np.ceil(im_w / self.size_divisor) * self.size_divisor)

        if h == im_h and w == im_w:
            return img.astype(np.float32)

        if self.pad_mode == -1:
            offset_x, offset_y = self.offsets
        elif self.pad_mode == 0:
            offset_y, offset_x = 0, 0
        elif self.pad_mode == 1:
            offset_y, offset_x = (h - im_h) // 2, (w - im_w) // 2
        else:
            offset_y, offset_x = h - im_h, w - im_w

        offsets, im_size, size = [offset_x, offset_y], [im_h, im_w], [h, w]

        return self.apply_image(img, offsets, im_size, size)


class RandomCropImage(object):
    """Random crop image only
    """

    def __init__(self, size):
        super(RandomCropImage, self).__init__()
        if isinstance(size, int):
            size = [size, size]
        self.size = size

    def __call__(self, img):

        h, w = img.shape[:2]
        tw, th = self.size
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)

        img = img[i:i + th, j:j + tw, :]
        return img


class RandCropImage(object):
    """ random crop image """

    def __init__(self,
                 size,
                 scale=None,
                 ratio=None,
                 interpolation=None,
                 backend="cv2"):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

        self._resize_func = UnifiedResize(
            interpolation=interpolation, backend=backend)

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]

        return self._resize_func(img, size)


class RandCropImageV2(object):
    """ RandCropImageV2 is different from RandCropImage,
    it will Select a cutting position randomly in a uniform distribution way,
    and cut according to the given size without resize at last."""

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img_h, img_w = img.shape[0], img.shape[1]
        else:
            img_w, img_h = img.size
        tw, th = self.size

        if img_h + 1 < th or img_w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".
                format((th, tw), (img_h, img_w)))

        if img_w == tw and img_h == th:
            return img

        top = random.randint(0, img_h - th + 1)
        left = random.randint(0, img_w - tw + 1)
        if isinstance(img, np.ndarray):
            return img[top:top + th, left:left + tw, :]
        else:
            return img.crop((left, top, left + tw, top + th))


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            if isinstance(img, np.ndarray):
                return cv2.flip(img, self.flip_code)
            else:
                return img.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return img


class AutoAugment(object):
    def __init__(self):
        self.policy = ImageNetPolicy()

    def __call__(self, img):
        from PIL import Image
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        img = self.policy(img)
        img = np.asarray(img)


class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)


class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))


class AugMix(object):
    """ Perform AugMix augmentation and compute mixture.
    """

    def __init__(self,
                 prob=0.5,
                 aug_prob_coeff=0.1,
                 mixture_width=3,
                 mixture_depth=1,
                 aug_severity=1):
        """
        Args:
            prob: Probability of taking augmix
            aug_prob_coeff: Probability distribution coefficients.
            mixture_width: Number of augmentation chains to mix per augmented example.
            mixture_depth: Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]'
            aug_severity: Severity of underlying augmentation operators (between 1 to 10).
        """
        # fmt: off
        self.prob = prob
        self.aug_prob_coeff = aug_prob_coeff
        self.mixture_width = mixture_width
        self.mixture_depth = mixture_depth
        self.aug_severity = aug_severity
        self.augmentations = augmentations
        # fmt: on

    def __call__(self, image):
        """Perform AugMix augmentations and compute mixture.
        Returns:
          mixed: Augmented and mixed image.
        """
        if random.random() > self.prob:
            # Avoid the warning: the given NumPy array is not writeable
            return np.asarray(image).copy()

        ws = np.float32(
            np.random.dirichlet([self.aug_prob_coeff] * self.mixture_width))
        m = np.float32(
            np.random.beta(self.aug_prob_coeff, self.aug_prob_coeff))

        # image = Image.fromarray(image)
        mix = np.zeros(image.shape)
        for i in range(self.mixture_width):
            image_aug = image.copy()
            image_aug = Image.fromarray(image_aug)
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(self.augmentations)
                image_aug = op(image_aug, self.aug_severity)
            mix += ws[i] * np.asarray(image_aug)

        mixed = (1 - m) * image + m * mix
        return mixed.astype(np.uint8)


class ColorJitter(RawColorJitter):
    """ColorJitter.
    """

    def __init__(self, prob=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob = prob

    def __call__(self, img):
        if np.random.random() < self.prob:
            if not isinstance(img, Image.Image):
                img = np.ascontiguousarray(img)
                img = Image.fromarray(img)
            img = super()._apply_image(img)
            if isinstance(img, Image.Image):
                img = np.asarray(img)
        return img


class Pad(object):
    """
    Pads the given PIL.Image on all sides with specified padding mode and fill value.
    adapted from: https://pytorch.org/vision/stable/_modules/torchvision/transforms/transforms.html#Pad
    """

    def __init__(self, padding: int, fill: int=0,
                 padding_mode: str="constant"):
        self.padding = padding
        self.fill = fill
        self.padding_mode = padding_mode

    def _parse_fill(self, fill, img, min_pil_version, name="fillcolor"):
        # Process fill color for affine transforms
        major_found, minor_found = (int(v)
                                    for v in PILLOW_VERSION.split('.')[:2])
        major_required, minor_required = (int(v) for v in
                                          min_pil_version.split('.')[:2])
        if major_found < major_required or (major_found == major_required and
                                            minor_found < minor_required):
            if fill is None:
                return {}
            else:
                msg = (
                    "The option to fill background area of the transformed image, "
                    "requires pillow>={}")
                raise RuntimeError(msg.format(min_pil_version))

        num_bands = len(img.getbands())
        if fill is None:
            fill = 0
        if isinstance(fill, (int, float)) and num_bands > 1:
            fill = tuple([fill] * num_bands)
        if isinstance(fill, (list, tuple)):
            if len(fill) != num_bands:
                msg = (
                    "The number of elements in 'fill' does not match the number of "
                    "bands of the image ({} != {})")
                raise ValueError(msg.format(len(fill), num_bands))

            fill = tuple(fill)

        return {name: fill}

    def __call__(self, img):
        opts = self._parse_fill(self.fill, img, "2.3.0", name="fill")
        if img.mode == "P":
            palette = img.getpalette()
            img = ImageOps.expand(img, border=self.padding, **opts)
            img.putpalette(palette)
            return img

        return ImageOps.expand(img, border=self.padding, **opts)
