#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import io
import tarfile
import numpy as np
from PIL import Image  #all use default backend

import paddle
from paddle.io import Dataset
import pickle
import os
import cv2
import random

from feature_extractor.data import preprocess
from feature_extractor.data.preprocess import transform
from feature_extractor.utils import logger


def create_operators(params):
    """
    create operators based on the config
    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(params, list), ('operator config should be a list')
    ops = []
    for operator in params:
        print(operator)
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        op = getattr(preprocess, op_name)(**param)
        ops.append(op)

    return ops


class ImageNetDataset(Dataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None, ):
        self._img_root = image_root
        self._cls_path = cls_label_path
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self._dtype = paddle.get_default_dtype()
        self._load_anno()

    def _load_anno(self):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        self.images = []
        self.labels = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for l in lines:
                l = l.strip().split(" ")
                self.images.append(os.path.join(self._img_root, l[0]))
                self.labels.append(int(l[1]))
                assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        try:
            img = cv2.imread(self.images[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[idx], img, self.labels[idx])
            #print(img.shape, self.labels[idx])
            #return {'image':img, 'label':self.labels[idx]}

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))


