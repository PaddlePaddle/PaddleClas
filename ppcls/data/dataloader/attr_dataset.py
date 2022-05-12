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
import pickle

from .common_dataset import CommonDataset
from ppcls.data.preprocess import transform


class AttrDataset(CommonDataset):
    def _load_anno(self, seed=None, split='trainval'):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)
        anno_path = self._cls_path
        image_dir = self._img_root
        self.images = []
        self.labels = []

        dataset_info = pickle.load(open(anno_path, 'rb+'))
        img_id = dataset_info.image_name

        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0
        attr_id = dataset_info.attr_name
        if 'label_idx' in dataset_info.keys():
            eval_attr_idx = dataset_info.label_idx.eval
            attr_label = attr_label[:, eval_attr_idx]
            attr_id = [attr_id[i] for i in eval_attr_idx]

        attr_num = len(attr_id)

        # mapping category name to class id
        # first_class:0, second_class:1, ...
        cname2cid = {attr_id[i]: i for i in range(attr_num)}

        assert split in dataset_info.partition.keys(
        ), f'split {split} is not exist'

        img_idx = dataset_info.partition[split]

        if isinstance(img_idx, list):
            img_idx = img_idx[0]  # default partition 0

        img_num = img_idx.shape[0]
        img_id = [img_id[i] for i in img_idx]
        label = attr_label[img_idx]  # [:, [0, 12]]
        self.label_ratio = label.mean(0)
        print("label_ratio:", self.label_ratio)
        for i, (img_i, label_i) in enumerate(zip(img_id, label)):
            imgname = os.path.join(image_dir, img_i)
            self.images.append(imgname)
            self.labels.append(np.int64(label_i))

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, [self.labels[idx], self.label_ratio])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
