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
# limitations under the License.from __future__ import division

# from ppcls.utils import logger
import nvidia.dali.fn as fn
import nvidia.dali.math as nvmath
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


class Pixels(ops.random.Normal):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 mode="const",
                 mean=[0.0, 0.0, 0.0],
                 channel_first=False,
                 h=224,
                 w=224,
                 c=3,
                 **kwargs):
        super(Pixels, self).__init__(*kargs, device=device, **kwargs)
        self._mode = mode
        self._mean = mean
        self.channel_first = channel_first
        self.h = h
        self.w = w
        self.c = c

    def __call__(self, **kwargs):
        if self._mode == "rand":
            return super(Pixels, self).__call__(shape=(
                3)) if not self.channel_first else super(
                    Pixels, self).__call__(shape=(3))
        elif self._mode == "pixel":
            return super(Pixels, self).__call__(shape=(
                self.h, self.w, self.c)) if not self.channel_first else super(
                    Pixels, self).__call__(shape=(self.c, self.h, self.w))
        elif self._mode == "const":
            return fn.constant(
                fdata=self._mean,
                shape=(self.c)) if not self.channel_first else fn.constant(
                    fdata=self._mean, shape=(self.c))
        else:
            raise Exception(
                "Invalid mode in RandomErasing, only support \"const\", \"rand\", \"pixel\""
            )


class RandomErasing(ops.Erase):
    def __init__(self,
                 *kargs,
                 device="cpu",
                 EPSILON=0.5,
                 sl=0.02,
                 sh=0.4,
                 r1=0.3,
                 mean=[0.0, 0.0, 0.0],
                 attempt=100,
                 use_log_aspect=False,
                 mode='const',
                 channel_first=False,
                 img_h=224,
                 img_w=224,
                 **kwargs):
        super(RandomErasing, self).__init__(*kargs, device=device, **kwargs)
        self.EPSILON = eval(EPSILON) if isinstance(EPSILON, str) else EPSILON
        self.sl = eval(sl) if isinstance(sl, str) else sl
        self.sh = eval(sh) if isinstance(sh, str) else sh
        r1 = eval(r1) if isinstance(r1, str) else r1
        self.r1 = (math.log(r1), math.log(1 / r1)) if use_log_aspect else (
            r1, 1 / r1)
        self.use_log_aspect = use_log_aspect
        self.attempt = attempt
        self.mean = mean
        self.get_pixels = Pixels(
            device=device,
            mode=mode,
            mean=mean,
            channel_first=False,
            h=224,
            w=224,
            c=3)
        self.channel_first = channel_first
        self.img_h = img_h
        self.img_w = img_w
        self.area = img_h * img_w

    def __call__(self, data, **kwargs):
        do_aug = fn.random.coin_flip(probability=self.EPSILON)
        keep = do_aug ^ True
        target_area = fn.random.uniform(range=(self.sl, self.sh)) * self.area
        aspect_ratio = fn.random.uniform(range=(self.r1[0], self.r1[1]))
        h = nvmath.floor(nvmath.sqrt(target_area * aspect_ratio))
        w = nvmath.floor(nvmath.sqrt(target_area / aspect_ratio))
        pixels = self.get_pixels()
        range1 = fn.stack(
            (self.img_h - h) / self.img_h - (self.img_h - h) / self.img_h,
            (self.img_h - h) / self.img_h)
        range2 = fn.stack(
            (self.img_w - w) / self.img_w - (self.img_w - w) / self.img_w,
            (self.img_w - w) / self.img_w)
        # shapes
        x1 = fn.random.uniform(range=range1)
        y1 = fn.random.uniform(range=range2)
        anchor = fn.stack(x1, y1)
        shape = fn.stack(h, w)
        aug_data = super(RandomErasing, self).__call__(
            data,
            anchor=anchor,
            normalized_anchor=True,
            shape=shape,
            fill_value=pixels)
        return fn.cast(
            aug_data * do_aug + data * keep, dtype=types.DALIDataType.UINT8)


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
        self.rng_angle = ops.random.Uniform(range=(-angle, angle))

    def __call__(self, data, **kwargs):
        do_rotate = self.rng()
        angle = self.rng_angle()
        flip_data = super(RandomRotation, self).__call__(
            data,
            angle=do_flip * angle,
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
        print(kwargs)
        super(NormalizeImage, self).__init__(*kargs, device=device, **kwargs)

    def __call__(self, data, **kwargs):
        return super(NormalizeImage, self).__call__(
            data, axes=[0, 1], **kwargs)
