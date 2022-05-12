#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import numpy as np
import os

from paddle.io import Dataset
from paddle.vision import transforms
import cv2
import warnings

from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.data.preprocess.ops.operators import DecodeImage
from ppcls.utils import logger
from ppcls.data.dataloader.common_dataset import create_operators


class MultiScaleDataset(Dataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None, ):
        self._img_root = image_root
        self._cls_path = cls_label_path
        self.transform_ops = transform_ops
        self.images = []
        self.labels = []
        self._load_anno()
        self.has_crop_flag = 1

    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip().split(" ")
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(np.int64(l[1]))
                assert os.path.exists(self.images[-1])

    def __getitem__(self, properties):
        # properites is a tuple, contains (width, height, index)
        img_width = properties[0]
        img_height = properties[1]
        index = properties[2]
        has_crop = False
        if self.transform_ops:
            for i in range(len(self.transform_ops)):
                op = self.transform_ops[i]
                resize_op = ['RandCropImage', 'ResizeImage', 'CropImage']
                for resize in resize_op:
                    if resize in op:
                        if self.has_crop_flag:
                            logger.warning(
                                "Multi scale dataset will crop image according to the multi scale resolution"
                            )
                        self.transform_ops[i][resize] = {
                            'size': (img_width, img_height)
                        }
                        has_crop = True
                        self.has_crop_flag = 0
        if has_crop == False:
            logger.error("Multi scale dateset requests RandCropImage")
            raise RuntimeError("Multi scale dateset requests RandCropImage")
        self._transform_ops = create_operators(self.transform_ops)

        try:
            with open(self.images[index], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[index])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[index], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))
