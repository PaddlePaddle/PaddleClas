from __future__ import print_function

import os
from typing import Callable, List

import numpy as np
import paddle
from paddle.io import Dataset
from PIL import Image
from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import create_operators


class GoodsDataset(Dataset):
    """Dataset for Goods, such as SOP, Inshop...

    Args:
        image_root (str): image root
        cls_label_path (str): path to annotation file
        transform_ops (List[Callable], optional): list of transform op(s). Defaults to None.
        backend (str, optional): pil or cv2. Defaults to "cv2".
        relabel (bool, optional): whether do relabel when original label do not starts from 0 or are discontinuous. Defaults to False.
    """

    def __init__(self,
                 image_root: str,
                 cls_label_path: str,
                 transform_ops: List[Callable]=None,
                 backend="cv2",
                 relabel=False):
        self._img_root = image_root
        self._cls_path = cls_label_path
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)
        self.backend = backend
        self._dtype = paddle.get_default_dtype()
        self._load_anno(relabel)

    def _load_anno(self, seed=None, relabel=False):
        assert os.path.exists(
            self._cls_path), f"path {self._cls_path} does not exist."
        assert os.path.exists(
            self._img_root), f"path {self._img_root} does not exist."
        self.images = []
        self.labels = []
        self.cameras = []
        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if relabel:
                label_set = set()
                for line in lines:
                    line = line.strip().split()
                    label_set.add(np.int64(line[1]))
                label_map = {
                    oldlabel: newlabel
                    for newlabel, oldlabel in enumerate(label_set)
                }

            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for line in lines:
                line = line.strip().split()
                self.images.append(os.path.join(self._img_root, line[0]))
                if relabel:
                    self.labels.append(label_map[np.int64(line[1])])
                else:
                    self.labels.append(np.int64(line[1]))
                self.cameras.append(np.int64(line[2]))
                assert os.path.exists(self.images[
                    -1]), f"path {self.images[-1]} does not exist."

    def __getitem__(self, idx):
        try:
            img = Image.open(self.images[idx]).convert("RGB")
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
