# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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
import os
from .common_dataset import CommonDataset, create_operators
from ..preprocess.ops.masking_generator import MaskingGenerator
import numpy as np
from ppcls.data.preprocess import transform

class BEiT_ImageNet(CommonDataset):
    cls_filter = None

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 patch_transforms=None,
                 visual_token_transforms=None,
                 masking_generator=None):
        super(BEiT_ImageNet, self).__init__(image_root, cls_label_path,
                                            transform_ops)

        self._patch_transform = create_operators(patch_transforms)
        self._visual_token_transform = create_operators(visual_token_transforms)
        self._masked_position_generator = MaskingGenerator(**masking_generator)

    def _load_anno(self):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            for line in lines:
                line = line.strip().split(" ")
                self.images.append(os.path.join(self._img_root, line[0]))
                self.labels.append(np.int64(line[1]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."

    def __getitem__(self, idx):
        with open(self.images[idx], 'rb') as f:
            img = f.read()
        for_patches, for_visual_tokens = transform(img, self._transform_ops)
        return \
            (transform(for_patches, self._patch_transform), \
            transform(for_visual_tokens, self._visual_token_transform), \
            self._masked_position_generator())
