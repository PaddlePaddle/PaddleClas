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

# normal_ops_1
decode_op = DecodeImage()
randcrop_op = RandCropImage(size=(size, size))

# trans_ops
autoaugment_op = ImageNetPolicy()
randaugment_op = RandAugment(3, 1)
cutout_op = Cutout()

# normal_ops_2
randflip_op = RandFlipImage(flip_code=1)
normalize_op = NormalizeImage(
    scale=img_scale, mean=img_mean, std=img_std, order='')
tochw_op = ToCHWImage()

# mask_ops
hide_and_seek_op = HideAndSeek()
randomerasing_op = RandomErasing()
gridmask_op = GridMask(d1=96, d2=224, rotate=360, ratio=0.6, mode=1, prob=0.8)

# batch_ops
mixup_op = MixupOperator(alpha=0.2)
cutmix_op = CutmixOperator(alpha=0.2)
fmix_op = FmixOperator()


def fakereader():
    """ fake reader """
    import random
    data = open(fname).read()

    def wrapper():
        while True:
            yield (data, random.randint(0, 1000))

    return wrapper


def superreader(batch_size=32):
    """ super reader """
    normal_ops_1 = [decode_op, randcrop_op]
    normal_ops_2 = [randflip_op, normalize_op, tochw_op]

    trans_ops = [autoaugment_op, randaugment_op, cutout_op]
    trans_ops_p = [0.2, 0.3, 0.5]
    mask_ops = [hide_and_seek_op, randomerasing_op, gridmask_op]
    mask_ops_p = [0.1, 0.6, 0.3]
    batch_ops = [mixup_op, cutmix_op, fmix_op]
    batch_ops_p = [0.3, 0.3, 0.4]

    reader = fakereader()

    def wrapper():
        batch = []
        for idx, sample in enumerate(reader()):
            img, label = sample
            ops = normal_ops_1 + [np.random.choice(trans_ops, p=trans_ops_p)] +\
                    normal_ops_2 + [np.random.choice(mask_ops, p=mask_ops_p)]
            img = transform(img, ops)
            batch.append((img, label))
            if (idx + 1) % batch_size == 0:
                batch = transform(
                    batch, [np.random.choice(
                        batch_ops, p=batch_ops_p)])
                yield batch
                batch = []

    return wrapper


if __name__ == '__main__':
    reader = superreader(32)
    for batch in reader():
        print(len(batch), len(batch[0]), batch[0][0].shape, batch[0][1:])
