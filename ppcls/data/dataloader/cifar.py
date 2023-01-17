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

from __future__ import print_function
import numpy as np
import cv2
from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.data.dataloader.common_dataset import create_operators
from paddle.vision.datasets import Cifar10 as Cifar10_paddle
from paddle.vision.datasets import Cifar100 as Cifar100_paddle


class Cifar10(Cifar10_paddle):
    def __init__(self,
                 data_file=None,
                 mode='train',
                 download=True,
                 backend='cv2',
                 sample_per_label=None,
                 expand_labels=1,
                 transform_ops=None,
                 transform_ops_weak=None,
                 transform_ops_strong=None,
                 transform_ops_strong2=None):
        super().__init__(data_file, mode, None, download, backend)
        assert isinstance(expand_labels, int)
        self._transform_ops = create_operators(transform_ops)
        self._transform_ops_weak = create_operators(transform_ops_weak)
        self._transform_ops_strong = create_operators(transform_ops_strong)
        self._transform_ops_strong2 = create_operators(transform_ops_strong2)
        self.class_num = 10
        labels = []
        for x in self.data:
            labels.append(x[1])
        labels = np.array(labels)
        if isinstance(sample_per_label, int):
            index = []
            for i in range(self.class_num):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, sample_per_label, False)
                index.extend(idx)
            index = index * expand_labels
            data = [self.data[x] for x in index]
            self.data = data

    def __getitem__(self, idx):
        (image, label) = super().__getitem__(idx)
        if self._transform_ops:
            image1 = transform(image, self._transform_ops)
            image1 = image1.transpose((2, 0, 1))
            return (image1, np.int64(label))
        elif self._transform_ops_weak and self._transform_ops_strong and self._transform_ops_strong2:
            image2 = transform(image, self._transform_ops_weak)
            image2 = image2.transpose((2, 0, 1))
            image3 = transform(image, self._transform_ops_strong)
            image3 = image3.transpose((2, 0, 1))
            image4 = transform(image, self._transform_ops_strong2)
            image4 = image4.transpose((2, 0, 1))
            return (image2, image3, image4, np.int64(label))

        elif self._transform_ops_weak and self._transform_ops_strong:
            image2 = transform(image, self._transform_ops_weak)
            image2 = image2.transpose((2, 0, 1))
            image3 = transform(image, self._transform_ops_strong)
            image3 = image3.transpose((2, 0, 1))

            return (image2, image3, np.int64(label))


class Cifar100(Cifar100_paddle):
    def __init__(self,
                 data_file=None,
                 mode='train',
                 download=True,
                 backend='pil',
                 sample_per_label=None,
                 expand_labels=1,
                 transform_ops=None,
                 transform_ops_weak=None,
                 transform_ops_strong=None,
                 transform_ops_strong2=None):
        super().__init__(data_file, mode, None, download, backend)
        assert isinstance(expand_labels, int)
        self._transform_ops = create_operators(transform_ops)
        self._transform_ops_weak = create_operators(transform_ops_weak)
        self._transform_ops_strong = create_operators(transform_ops_strong)
        self._transform_ops_strong2 = create_operators(transform_ops_strong2)
        self.class_num = 100

        labels = []
        for x in self.data:
            labels.append(x[1])
        labels = np.array(labels)
        if isinstance(sample_per_label, int):
            index = []
            for i in range(self.class_num):
                idx = np.where(labels == i)[0]
                idx = np.random.choice(idx, sample_per_label, False)
                index.extend(idx)
            index = index * expand_labels
            data = [self.data[x] for x in index]
            self.data = data

    def __getitem__(self, idx):
        (image, label) = super().__getitem__(idx)
        if self._transform_ops:
            image1 = transform(image, self._transform_ops)
            image1 = image1.transpose((2, 0, 1))
            return (image1, np.int64(label))
        elif self._transform_ops_weak and self._transform_ops_strong and self._transform_ops_strong2:
            image2 = transform(image, self._transform_ops_weak)
            image2 = image2.transpose((2, 0, 1))
            image3 = transform(image, self._transform_ops_strong)
            image3 = image3.transpose((2, 0, 1))
            image4 = transform(image, self._transform_ops_strong2)
            image4 = image4.transpose((2, 0, 1))
            return (image2, image3, image4, np.int64(label))
        elif self._transform_ops_weak and self._transform_ops_strong:
            image2 = transform(image, self._transform_ops_weak)
            image2 = image2.transpose((2, 0, 1))
            image3 = transform(image, self._transform_ops_strong)
            image3 = image3.transpose((2, 0, 1))

            return (image2, image3, np.int64(label))