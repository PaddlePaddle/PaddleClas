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

from __future__ import division

import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class DecodeImage(ops.decoders.Image):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(DecodeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(DecodeImage, self).__call__(data, **kwargs)


class ToCHWImage(ops.Transpose):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(ToCHWImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(ToCHWImage, self).__call__(data, **kwargs)


class ColorJitter(ops.ColorTwist):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 prob=1.0,
                 brightness_factor=0.0,
                 contrast_factor=0.0,
                 saturation_factor=0.0,
                 hue_factor=0.0,
                 **kwargs):
        super(ColorJitter, self).__init__(*kargs, device=device, **kwargs)
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.saturation_factor = saturation_factor
        self.hue_factor = hue_factor
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_jitter = self.rng()
        brightness = fn.random.uniform(
            range=(max(0, 1 - self.brightness_factor),
                   1 + self.brightness_factor)) * do_jitter
        contrast = fn.random.uniform(
            range=(max(0, 1 - self.contrast_factor),
                   1 + self.contrast_factor)) * do_jitter
        saturation = fn.random.uniform(
            range=(max(0, 1 - self.saturation_factor),
                   1 + self.saturation_factor)) * do_jitter
        hue = fn.random.uniform(range=(-self.hue_factor,
                                       self.hue_factor)) * do_jitter
        return super(ColorJitter, self).__call__(
            data,
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue,
            **kwargs)


class DecodeRandomResizedCrop(ops.decoders.ImageRandomCrop):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 resize_x=224,
                 resize_y=224,
                 resize_short=None,
                 interp_type=types.DALIInterpType.INTERP_LINEAR,
                 **kwargs):
        super(DecodeRandomResizedCrop, self).__init__(
            *kargs, device=device, **kwargs)
        if resize_short is None:
            self.resize = ops.Resize(
                device="gpu" if device == "mixed" else "cpu",
                resize_x=resize_x,
                resize_y=resize_y,
                interp_type=interp_type)
        else:
            self.resize = ops.Resize(
                device="gpu" if device == "mixed" else "cpu",
                resize_short=resize_short,
                interp_type=interp_type)

    def __call__(self, data, **kwargs):
        data = super(DecodeRandomResizedCrop, self).__call__(data, **kwargs)
        data = self.resize(data)
        return data


class CropMirrorNormalize(ops.CropMirrorNormalize):
    def __init__(self, *kargs, device="cpu", prob=0.5, **kwargs):
        super(CropMirrorNormalize, self).__init__(
            *kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_mirror = self.rng()
        return super(CropMirrorNormalize, self).__call__(
            data, mirror=do_mirror, **kwargs)


class RandCropImage(ops.RandomResizedCrop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandCropImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(RandCropImage, self).__call__(data, **kwargs)


class CropImage(ops.Crop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(CropImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(CropImage, self).__call__(data, **kwargs)


class ResizeImage(ops.Resize):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(ResizeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(ResizeImage, self).__call__(data, **kwargs)


class RandFlipImage(ops.Flip):
    def __init__(self, *kargs, device="cpu", prob=0.5, flip_code=1, **kwargs):
        super(RandFlipImage, self).__init__(*kargs, device=device, **kwargs)
        self.flip_code = flip_code
        self.rng = ops.random.CoinFlip(probability=prob)

    def __call__(self, data, **kwargs):
        do_flip = self.rng()
        if self.flip_code == 1:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=0, **kwargs)
        elif self.flip_code == 0:
            return super(RandFlipImage, self).__call__(
                data, horizontal=0, vertical=do_flip, **kwargs)
        else:
            return super(RandFlipImage, self).__call__(
                data, horizontal=do_flip, vertical=do_flip, **kwargs)


class Pad(ops.Crop):
    """
    use ops.Crop to implement Pad operator, for ops.Pad alwayls only pad in right and bottom.
    """

    def __init__(self, *kargs, device="cpu", **kwargs):
        super(Pad, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(Pad, self).__call__(data, **kwargs)


class RandCropImageV2(ops.Crop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandCropImageV2, self).__init__(*kargs, device=device, **kwargs)
        self.rng_x = ops.random.Uniform(range=(0.0, 1.0))
        self.rng_y = ops.random.Uniform(range=(0.0, 1.0))

    def __call__(self, data, **kwargs):
        pos_x = self.rng_x()
        pos_y = self.rng_y()
        return super(RandCropImageV2, self).__call__(
            data, crop_pos_x=pos_x, crop_pos_y=pos_y, **kwargs)


class RandomCropImage(ops.Crop):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandomCropImage, self).__init__(*kargs, device=device, **kwargs)
        self.rng_x = ops.random.Uniform(range=(0.0, 1.0))
        self.rng_y = ops.random.Uniform(range=(0.0, 1.0))

    def __call__(self, data, **kwargs):
        pos_x = self.rng_x()
        pos_y = self.rng_y()
        return super(RandomCropImage, self).__call__(
            data, crop_pos_x=pos_x, crop_pos_y=pos_y, **kwargs)


class RandomRotation(ops.Rotate):
    def __init__(self, *kargs, device="cpu", prob=0.5, angle=0, **kwargs):
        super(RandomRotation, self).__init__(*kargs, device=device, **kwargs)
        self.rng = ops.random.CoinFlip(probability=prob)
        discrete_angle = list(range(-angle, angle + 1))
        self.rng_angle = ops.random.Uniform(values=discrete_angle)

    def __call__(self, data, **kwargs):
        do_rotate = self.rng()
        angle = self.rng_angle()
        flip_data = super(RandomRotation, self).__call__(
            data,
            angle=do_rotate * angle,
            keep_size=True,
            fill_value=0,
            **kwargs)
        return flip_data


class RandomRot90(ops.Rotate):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(RandomRot90, self).__init__(*kargs, device=device, **kwargs)
        self.rng_angle = ops.random.Uniform(values=[0.0, 1.0, 2.0, 3.0])

    def __call__(self, data, **kwargs):
        angle = self.rng_angle() * 90.0
        flip_data = super(RandomRot90, self).__call__(
            data, angle=angle, keep_size=True, fill_value=0, **kwargs)
        return flip_data


class NormalizeImage(ops.Normalize):
    def __init__(self, *kargs, device="cpu", **kwargs):
        super(NormalizeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(NormalizeImage, self).__call__(data, **kwargs)
