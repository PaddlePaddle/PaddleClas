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

import paddle
from paddle.vision.datasets import Cifar10
from paddle.vision import transforms
from paddle.dataset.common import _check_exists_and_download

import numpy as np
import os
from PIL import Image


class CustomizedCifar10(Cifar10):
    def __init__(self,
                 data_file=None,
                 mode='train',
                 download=True,
                 backend=None):
        assert mode.lower() in ['train', 'test', 'train', 'test'], \
            "mode should be 'train10', 'test10', 'train100' or 'test100', but got {}".format(mode)
        self.mode = mode.lower()

        if backend is None:
            backend = paddle.vision.get_image_backend()
        if backend not in ['pil', 'cv2']:
            raise ValueError(
                "Expected backend are one of ['pil', 'cv2'], but got {}"
                .format(backend))
        self.backend = backend

        self._init_url_md5_flag()

        self.data_file = data_file
        if self.data_file is None:
            assert download, "data_file is not set and downloading automatically is disabled"
            self.data_file = _check_exists_and_download(
                data_file, self.data_url, self.data_md5, 'cifar', download)

        self.transform = transforms.Compose([
            transforms.Resize(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._load_data()
        self.dtype = paddle.get_default_dtype()

    def __getitem__(self, index):
        img, target = self.data[index]
        img = np.reshape(img, [3, 32, 32])
        img = img.transpose([1, 2, 0]).astype("uint8")
        img = Image.fromarray(img)
        img = self.transform(img)
        return (img, target)
