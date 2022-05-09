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
import paddle
from paddle.io import Dataset
import os
import cv2

from ppcls.data import preprocess
from ppcls.data.preprocess import transform
from ppcls.utils import logger
from .common_dataset import create_operators
import os.path as osp
import glob
import re
from PIL import Image


class Market1501(Dataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    _dataset_dir = 'market1501/Market-1501-v15.09.15'

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 backend="cv2"):
        self._img_root = image_root
        self._cls_path = cls_label_path  # the sub folder in the dataset
        self._dataset_dir = osp.join(image_root, self._dataset_dir,
                                     self._cls_path)
        self._check_before_run()
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self.backend = backend
        self._dtype = paddle.get_default_dtype()
        self._load_anno(relabel=True if 'train' in self._cls_path else False)

    def _check_before_run(self):
        """Check if the file is available before going deeper"""
        if not osp.exists(self._dataset_dir):
            raise RuntimeError("'{}' is not available".format(
                self._dataset_dir))

    def _load_anno(self, relabel=False):
        img_paths = glob.glob(osp.join(self._dataset_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        self.images = []
        self.labels = []
        self.cameras = []
        pid_container = set()

        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            self.images.append(img_path)
            self.labels.append(pid)
            self.cameras.append(camid)

        self.num_pids, self.num_imgs, self.num_cams = get_imagedata_info(
            self.images, self.labels, self.cameras, subfolder=self._cls_path)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            if self.backend == "cv2":
                img = np.array(img, dtype="float32").astype(np.uint8)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            if self.backend == "cv2":
                img = img.transpose((2, 0, 1))
            return (img, self.labels[idx], self.cameras[idx])
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


class MSMT17(Dataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    _dataset_dir = 'msmt17/MSMT17_V1'

    def __init__(self, image_root, cls_label_path, transform_ops=None):
        self._img_root = image_root
        self._cls_path = cls_label_path  # the sub folder in the dataset
        self._dataset_dir = osp.join(image_root, self._dataset_dir,
                                     self._cls_path)
        self._check_before_run()
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self._dtype = paddle.get_default_dtype()
        self._load_anno(relabel=True if 'train' in self._cls_path else False)

    def _check_before_run(self):
        """Check if the file is available before going deeper"""
        if not osp.exists(self._dataset_dir):
            raise RuntimeError("'{}' is not available".format(
                self._dataset_dir))

    def _load_anno(self, relabel=False):
        img_paths = glob.glob(osp.join(self._dataset_dir, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        self.images = []
        self.labels = []
        self.cameras = []
        pid_container = set()

        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 1 <= camid <= 15
            camid -= 1  # index starts from 0
            if relabel:
                pid = pid2label[pid]
            self.images.append(img_path)
            self.labels.append(pid)
            self.cameras.append(camid)

        self.num_pids, self.num_imgs, self.num_cams = get_imagedata_info(
            self.images, self.labels, self.cameras, subfolder=self._cls_path)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert('RGB')
            img = np.array(img, dtype="float32").astype(np.uint8)
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[idx], self.cameras[idx])
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


def get_imagedata_info(data, labels, cameras, subfolder='train'):
    pids, cams = [], []
    for _, pid, camid in zip(data, labels, cameras):
        pids += [pid]
        cams += [camid]
    pids = set(pids)
    cams = set(cams)
    num_pids = len(pids)
    num_cams = len(cams)
    num_imgs = len(data)
    print("Dataset statistics:")
    print("  ----------------------------------------")
    print("  subset   | # ids | # images | # cameras")
    print("  ----------------------------------------")
    print("  {}    | {:5d} | {:8d} | {:9d}".format(subfolder, num_pids,
                                                   num_imgs, num_cams))
    print("  ----------------------------------------")
    return num_pids, num_imgs, num_cams
