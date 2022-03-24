from __future__ import print_function

import numpy as np
from paddle.vision.datasets import MNIST
from ppcls.data.preprocess import transform
from ppcls.utils import logger

from .common_dataset import create_operators


class MnistDataset(MNIST):
    def __init__(self, transform_ops=None, **kwargs):
        super(MnistDataset, self).__init__(**kwargs)
        if transform_ops:
            self._transform_ops = create_operators(transform_ops)

    def __getitem__(self, idx):
        try:
            img = self.images[idx]
            img = np.reshape(img, [28, 28, 1])
            if self._transform_ops:
                img = transform(img, self._transform_ops)
            img = img.transpose((2, 0, 1))
            return (img, self.labels[idx])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    def __len__(self):
        return len(self.images)

    @property
    def class_num(self):
        return len(set(np.array(self.labels).flatten().tolist()))
