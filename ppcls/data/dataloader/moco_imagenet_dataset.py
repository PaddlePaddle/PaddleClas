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
from ppcls.utils import logger
from .common_dataset import CommonDataset, create_operators
from ppcls.data.preprocess import transform


class MoCoImageNetDataset(CommonDataset):
    """MoCoImageNetDataset

    Args:
        image_root (str): image root, path to `ILSVRC2012`
        cls_label_path (str): path to annotation file `train_list.txt` or `val_list.txt`
        return_label (bool, optional): whether return original label.
        return_two_sample (bool, optional): whether return two views about original image.
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
        view_trans1 (list): some transform op(s) for view1.
        view_trans2 (list): some transform op(s) for view2.
    """

    def __init__(
            self,
            image_root,
            cls_label_path,
            return_label=True,
            return_two_sample=False,
            transform_ops=None,
            delimiter=None,
            relabel=False,
            view_trans1=None,
            view_trans2=None, ):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        super(MoCoImageNetDataset, self).__init__(image_root, cls_label_path,
                                                  transform_ops)

        self.return_label = return_label
        self.return_two_sample = return_two_sample

        if self.return_two_sample:
            self.view_transform1 = create_operators(view_trans1)
            self.view_transform2 = create_operators(view_trans2)

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()

            if self.return_two_sample:
                sample1 = transform(img, self._transform_ops)
                sample2 = transform(img, self._transform_ops)
                sample1 = transform(sample1, self.view_transform1)
                sample2 = transform(sample2, self.view_transform2)

                if self.return_label:
                    return (sample1, sample2, self.labels[idx])
                else:
                    return (sample1, sample2)

            if self._transform_ops:
                img = transform(img, self._transform_ops)
                img = img.transpose((2, 0, 1))

            return (img, self.labels[idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if self.relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split(self.delimiter)
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(label_set)
                }

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split(self.delimiter)
                self.images.append(os.path.join(self._img_root, line[0]))
                if self.relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."
