#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import random
import math
from .common_dataset import CommonDataset
# from paddle.vision.datasets import ImageFolder

class IBOTDataset(CommonDataset):
    """ImageNetDataset
    Args:
        image_root (str): image root, path to `ILSVRC2012`
        cls_label_path (str): path to annotation file `train_list.txt` or `val_list.txt`
        transform_ops (list, optional): list of transform op(s). Defaults to None.
        delimiter (str, optional): delimiter. Defaults to None.
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
    """

    def __init__(self,
                 image_root,
                 cls_label_path,
                 transform_ops=None,
                 delimiter=None,
                 relabel=False,
                 patch_size=16,
                 pred_ratio=0.3,
                 pred_ratio_var=0,
                 pred_aspect_ratio=(0.3, 1/0.3),
                 pred_shape='block',
                 pred_start_epoch=0
                 ):
        self.delimiter = delimiter if delimiter is not None else " "
        self.relabel = relabel
        super(IBOTDataset, self).__init__(image_root, cls_label_path,
                                              transform_ops)
        self.psz = patch_size
        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and \
                                           len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and \
                                                   len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)
        self.log_aspect_ratio = tuple(map(lambda x: math.log(x), pred_aspect_ratio))
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

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

    def set_epoch(self, epoch):
        self.epoch = epoch

    def get_pred_ratio(self):
        if hasattr(self, 'epoch') and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = [
                random.uniform(prm - prv, prm + prv) if prv > 0 and prm >= prv else prm
                for prm, prv in zip(self.pred_ratio, self.pred_ratio_var)
            ]
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var, self.pred_ratio + \
                                        self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio

        return pred_ratio

    def __getitem__(self, idx):
        output = super(IBOTDataset, self).__getitem__(idx)

        masks = []
        for img in output[0]:
            try:
                H, W = img.shape[1] // self.psz, img.shape[2] // self.psz
            except:
                # skip non-image
                continue

            high = self.get_pred_ratio() * H * W
            if self.pred_shape == 'block':
                # following BEiT (https://arxiv.org/abs/2106.08254), see at
                # https://github.com/microsoft/unilm/blob/b94ec76c36f02fb2b0bf0dcb0b8554a2185173cd/beit/masking_generator.py#L55
                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < high:
                    max_mask_patches = high - mask_count

                    delta = 0
                    for attempt in range(10):
                        low = (min(H, W) // 3) ** 2
                        target_area = random.uniform(low, max_mask_patches)
                        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                        h = int(round(math.sqrt(target_area * aspect_ratio)))
                        w = int(round(math.sqrt(target_area / aspect_ratio)))
                        if w < W and h < H:
                            top = random.randint(0, H - h)
                            left = random.randint(0, W - w)

                            num_masked = mask[top: top + h, left: left + w].sum()
                            if 0 < h * w - num_masked <= max_mask_patches:
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta

            elif self.pred_shape == 'rand':
                mask = np.hstack([
                    np.zeros(H * W - int(high)),
                    np.ones(int(high)),
                ]).astype(bool)
                np.random.shuffle(mask)
                mask = mask.reshape(H, W)
            else:
                raise ValueError("Invalid pred shape you input it.")

            masks.append(mask)

        return output + (masks,)