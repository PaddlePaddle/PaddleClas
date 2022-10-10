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

import os
import numpy as np

from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import CommonDataset


class CustomLabelDataset(CommonDataset):
    """CustomLabelDataset

    Args:
        image_root (str): image root, path to `ILSVRC2012`
        sample_list_path (str): path to the file with samples listed.
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        label_key (str, optional): Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
    """

    def __init__(self,
                 image_root,
                 sample_list_path,
                 transform_ops=None,
                 label_key=None,
                 delimiter=None):
        self.delimiter = delimiter
        super().__init__(image_root, sample_list_path, transform_ops)
        if self._transform_ops is None and label_key is not None:
            label_key = None
            msg = "Unable to get label by label_key when transform_ops is None. The label_key has been set to None."
            logger.warning(msg)
        self.label_key = label_key

    def _load_anno(self, seed=None):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip()
                if self.delimiter is not None:
                    line = line.split(self.delimiter)[0]
                self.images.append(os.path.join(self._img_root, line))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                processed_sample = transform({"img": img}, self._transform_ops)
                img = processed_sample["img"].transpose((2, 0, 1))
                if self.label_key is not None:
                    label = processed_sample[self.label_key]
                    sample = (img, label)
                    return sample
            return (img)

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)
