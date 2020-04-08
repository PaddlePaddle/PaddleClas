# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from ppcls.data.imaug import DecodeImage
from ppcls.data.imaug import ResizeImage
from ppcls.data.imaug import RandCropImage
from ppcls.data.imaug import RandFlipImage
from ppcls.data.imaug import NormalizeImage
from ppcls.data.imaug import ToCHWImage

from ppcls.data.imaug import ImageNetPolicy
from ppcls.data.imaug import RandAugment
from ppcls.data.imaug import Cutout

from ppcls.data.imaug import HideAndSeek
from ppcls.data.imaug import RandomErasing
from ppcls.data.imaug import GridMask

from ppcls.data.imaug import MixupOperator
from ppcls.data.imaug import CutmixOperator
from ppcls.data.imaug import FmixOperator

from ppcls.data.imaug import transform

import numpy as np

fname = './test/demo.jpeg'
size = 224
img_mean = [0.485, 0.456, 0.406]
img_std = [0.229, 0.224, 0.225]
img_scale = 1.0 / 255.0

decode_op = DecodeImage()
randcrop_op = RandCropImage(size=(size, size))
randflip_op = RandFlipImage(flip_code=1)

normalize_op = NormalizeImage(
    scale=img_scale, mean=img_mean, std=img_std, order='')
tochw_op = ToCHWImage()

data = open(fname).read()


def print_function_name(func):
    """ print function name"""

    def wrapper(*args, **kwargs):
        """ wrapper """
        print("========Test Fuction: [%s]:" % (func.__name__))
        func(*args, **kwargs)
        print("========Test Fuction: [%s] done!\n" % (func.__name__))

    return wrapper


@print_function_name
def test_decode():
    """ test decode operator """
    img = decode_op(data)
    print('img shape is %s' % (str(img.shape)))


@print_function_name
def test_randcrop():
    """ test randcrop operator """
    img = decode_op(data)
    img = randcrop_op(img)
    assert img.shape == (size, size, 3), \
            'image shape[%s] should be equal to [%s]' % (img.shape, (size, size, 3))


@print_function_name
def test_randflip():
    """ test randflip operator """
    import cv2
    img = transform(data, [decode_op, randcrop_op])
    for i in xrange(10):
        flip_img = randflip_op(img)
        if np.array_equal(cv2.flip(img, 1), flip_img):
            break
    assert np.array_equal(cv2.flip(img, 1),
                          flip_img), 'you should check randcrop operator'


@print_function_name
def test_normalize():
    """ test normalize operator """
    img = transform(data, [decode_op, randcrop_op])

    norm_img = normalize_op(img)
    assert norm_img.dtype == np.float32, 'img.dtype should be float32 after normalizing'
    assert norm_img.shape == (size, size, 3), \
            'image shape[%s] should be equal to [%s]' % (norm_img.shape, (size, size, 3))
    print('max value of the img after normalizing is : %f' %
          (np.max(norm_img.flatten())))
    print('min value of the img after normalizing is : %f' %
          (np.min(norm_img.flatten())))


@print_function_name
def test_tochw():
    """ test  tochw operator """
    img = transform(data, [decode_op, randcrop_op, randflip_op, normalize_op])

    tochw_img = tochw_op(img)
    assert tochw_img.dtype == np.float32, 'img.dtype should be float32 after tochw'
    assert tochw_img.shape == (3, size, size), \
            'image shape[%s] should be equal to [%s]' % (tochw_img.shape, (3, size, size))


@print_function_name
def test_autoaugment():
    """ test autoaugment operator """
    from PIL import Image
    autoaugment_op = ImageNetPolicy()
    img = transform(data, [decode_op, randcrop_op])

    aa_img = autoaugment_op(img)
    assert aa_img.dtype == np.uint8, 'img.dtype should be uint8 after autoaugment'
    assert aa_img.shape == (size, size, 3), \
            'image shape[%s] should be equal to [%s]' % (aa_img.shape, (size, size, 3))


@print_function_name
def test_randaugment():
    """ test randaugment operator """
    from PIL import Image
    randaugment_op = RandAugment(3, 1)
    img = transform(data, [decode_op, randcrop_op])

    ra_img = randaugment_op(img)
    assert ra_img.dtype == np.uint8, 'img.dtype should be uint8 after randaugment'
    assert ra_img.shape == (size, size, 3), \
            'image shape[%s] should be equal to [%s]' % (ra_img.shape, (size, size, 3))


@print_function_name
def test_cutout():
    """ test cutout operator """
    cutout_op = Cutout()
    img = transform(data, [decode_op, randcrop_op])

    cutout_img = cutout_op(img)
    assert cutout_img.dtype == np.uint8, 'img.dtype should be uint8 after cutout'
    assert cutout_img.shape == (size, size, 3), \
            'image shape[%s] should be equal to [%s]' % (cutout_img.shape, (size, size, 3))


@print_function_name
def test_hideandseek():
    """ test hide and seek operator """
    img = transform(
        data, [decode_op, randcrop_op, randflip_op, normalize_op, tochw_op])

    hide_and_seek_op = HideAndSeek()
    hs_img = hide_and_seek_op(img)
    assert hs_img.dtype == np.float32, 'img.dtype should be float32 after hide and seek'
    assert hs_img.shape == (3, size, size), \
            'image shape[%s] should be equal to [%s]' % (hs_img.shape, (3, size, size))


@print_function_name
def test_randerasing():
    """ test randerasing operator """
    img = transform(
        data, [decode_op, randcrop_op, randflip_op, normalize_op, tochw_op])

    randomerasing_op = RandomErasing()
    re_img = randomerasing_op(img)
    assert re_img.dtype == np.float32, 'img.dtype should be float32 after randomerasing'
    assert re_img.shape == (3, size, size), \
            'image shape[%s] should be equal to [%s]' % (re_img.shape, (3, size, size))


@print_function_name
def test_gridmask():
    """ test gridmask operator """
    img = transform(
        data, [decode_op, randcrop_op, randflip_op, normalize_op, tochw_op])

    gridmask_op = GridMask(
        d1=96, d2=224, rotate=360, ratio=0.6, mode=1, prob=0.8)
    gm_img = gridmask_op(img)
    assert gm_img.dtype == np.float32, 'img.dtype should be float32 after gridmask'
    assert gm_img.shape == (3, size, size), \
            'image shape[%s] should be equal to [%s]' % (gr_img.shape, (3, size, size))


def generate_batch(batch_size=32):
    """ generate_batch """
    import random
    ops = [decode_op, randcrop_op, randflip_op, normalize_op, tochw_op]
    batch = [(transform(data, ops), random.randint(0, 1000))
             for i in xrange(batch_size)]
    return batch


def test_batch_operator(operator, batch_size):
    """ test batch operator """
    batch = generate_batch(batch_size)
    assert len(batch) == batch_size, \
            'num of samples not equal to batch_size: %d != %d' % (len(batch), batch_size)

    assert len(batch[0]) == 2, \
            'length of sample not equal to 2: %d != 2' % (len(batch[0]))

    import time
    tic = time.time()
    new_batch = operator(batch)
    cost = time.time() - tic
    print("operator cost: %.4fms" % (cost * 1000))

    assert len(batch) == len(new_batch), \
            'num of samples not equal: %d != %d' % (len(batch), len(new_batch))
    assert len(new_batch[0]) == 4, \
            'length of sample not equal to 4: %d != 4' % (len(new_batch[0]))


@print_function_name
def test_mixup():
    """ test mixup operator """
    batch_size = 32
    mixup_op = MixupOperator(alpha=0.2)
    test_batch_operator(mixup_op, batch_size)


@print_function_name
def test_cutmix():
    """ test cutmix operator """
    batch_size = 32
    cutmix_op = CutmixOperator(alpha=0.2)
    test_batch_operator(cutmix_op, batch_size)


@print_function_name
def test_fmix():
    """ test fmix operator """
    batch_size = 32
    fmix_op = FmixOperator()
    test_batch_operator(fmix_op, batch_size)


if __name__ == '__main__':
    test_decode()
    test_randcrop()
    test_randflip()
    test_normalize()
    test_tochw()

    test_autoaugment()
    test_randaugment()
    test_cutout()

    test_hideandseek()
    test_randerasing()
    test_gridmask()

    test_mixup()
    test_cutmix()
    test_fmix()
