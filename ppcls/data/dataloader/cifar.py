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
import shutil
from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.data.preprocess import BaseTransform, ListTransform
from ppcls.data.dataloader.common_dataset import create_operators
from paddle.vision.datasets import Cifar10 as Cifar10_paddle
from paddle.vision.datasets import Cifar100 as Cifar100_paddle
from paddle.vision.datasets import cifar
import os
from PIL import Image


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
                 transform_ops_strong=None):
        super().__init__(data_file, mode, None, download, backend)
        assert isinstance(expand_labels, int)
        self._transform_ops = create_operators(transform_ops)
        self._transform_ops_weak = create_operators(transform_ops_weak)
        self._transform_ops_strong = create_operators(transform_ops_strong)
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
                 transform_ops_strong=None):
        super().__init__(data_file, mode, None, download, backend)
        assert isinstance(expand_labels, int)
        self._transform_ops = create_operators(transform_ops)
        self._transform_ops_weak = create_operators(transform_ops_weak)
        self._transform_ops_strong = create_operators(transform_ops_strong)
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
        elif self._transform_ops_weak and self._transform_ops_strong:
            image2 = transform(image, self._transform_ops_weak)
            image2 = image2.transpose((2, 0, 1))
            image3 = transform(image, self._transform_ops_strong)
            image3 = image3.transpose((2, 0, 1))

            return (image2, image3, np.int64(label))


def np_convert_pil(array):
    """
    array conver image
    Args:
        array: array and dim is 1
    """
    assert len(array.shape), "dim of array should 1"
    img = Image.fromarray(array.reshape(3, 32, 32).transpose(1, 2, 0))
    return img


class CIFAR10(cifar.Cifar10):
    """
    cifar10 dataset
    """
    def __init__(self, data_file, download=True, mode='train'):
        super().__init__(download=download, mode=mode)
        if data_file is not None:
            os.makedirs(data_file, exist_ok=True)
            if not os.path.exists(os.path.join(data_file, 'cifar-10-python.tar.gz')):
                shutil.move('~/.cache/paddle/dataset/cifar/cifar-10-python.tar.gz', data_file)
        self.num_classes = 10
        self.x = []
        self.y = []
        for d in self.data:
            self.x.append(d[0])
            self.y.append(d[1])

        self.x = np.array(self.x)
        self.y = np.array(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.x.shape[0]


class CIFAR10SSL(CIFAR10):
    """
    from Cifar10
    """

    def __init__(self, 
                 data_file=None, 
                 sample_per_label=None, 
                 download=True,
                 expand_labels=1,
                 mode='train',
                 transform_ops=None,
                 transform_w=None,
                 transform_s1=None,
                 transform_s2=None):
        super().__init__(data_file, download=download, mode=mode)
        self.data_type = 'unlabeled_train' if mode == 'train' else 'val'
        if transform_ops is not None and sample_per_label is not None:
            index = []
            self.data_type = 'labeled_train'
            for c in range(self.num_classes):
                idx = np.where(self.y == c)[0]
                idx = np.random.choice(idx, sample_per_label, False)
                index.extend(idx)
            index = index * expand_labels
            self.x = self.x[index]
            self.y = self.y[index]
        self.transforms = [transform_ops] if transform_ops is not None else [transform_w, transform_s1, transform_s2]
        self.mode = mode

    def __getitem__(self, idx):
        img, label = np_convert_pil(self.x[idx]), self.y[idx]
        results = ListTransform(self.transforms)(img)
        if self.data_type == 'unlabeled_train':
            return results
        return results[0], label
        
    def __len__(self):
        return self.x.shape[0]
        

# def x_u_split(num_labeled, num_classes, label):
#     """
#     split index of dataset to labeled x and unlabeled u
#     Args:
#         num_labeled: num of labeled dataset
#         label: list or array, label
#     """
#     assert num_labeled <= len(label), "arg num_labeled should <= num of label"
#     label = np.array(label) if isinstance(label, list) else label
#     label_per_class = num_labeled // num_classes
#     labeled_idx = []
#     unlabeled_idx = np.array(list(range(label.shape[0])))
#     for c in range(num_classes):
#         idx = np.where(label == c)[0]
#         idx = np.random.choice(idx, label_per_class, False)
#         labeled_idx.extend(idx)
    
#     np.random.shuffle(labeled_idx)
#     return labeled_idx, unlabeled_idx