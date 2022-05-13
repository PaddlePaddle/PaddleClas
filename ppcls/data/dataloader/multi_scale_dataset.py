#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import os

import numpy as np
import cv2

from paddle.io import Dataset

from ppcls.data.preprocess import transform
from ppcls.data.dataloader.common_dataset import create_operators
from ppcls.utils import logger


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
        self.has_logged = False

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

    def __getitem__(self, batch_info):
        width, height, img_idx = batch_info
        if self.transform_ops:
            # TODO(gaotingquan): unsupport ones of 'RandCropImage', 'ResizeImage', 'CropImage' used together.
            for op in self.transform_ops:
                op_name = list(op.keys())[0]
                resize_ops = ['RandCropImage', 'ResizeImage', 'CropImage']
                if op_name in resize_ops:
                    op[op_name].update({"size": (width, height)})
                    has_changed = True
                    break
        # TODO(gaotingquan): repeat log
        if not self.has_logged:
            if has_changed == False:
                msg = "One of 'RandCropImage', 'ResizeImage', 'CropImage' should be use in MultiScale Dataset when MultiScale Sampler used. Otherwise the multi scale resolution strategy is ineffective."
            else:
                msg = f"The resize argument of '{op_name}' has been reset to {width}, {height} according to MultiScale Sampler."
            logger.warning(msg)
            self.has_logged = True

        self._transform_ops = create_operators(self.transform_ops)

        try:
            with open(self.images[img_idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[img_idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[img_idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(self.labels))
