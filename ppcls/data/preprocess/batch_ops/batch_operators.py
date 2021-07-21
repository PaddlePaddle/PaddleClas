# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np

from ppcls.data.preprocess.ops.fmix import sample_mask


class BatchOperator(object):
    """ BatchOperator """

    def __init__(self, *args, **kwargs):
        pass

    def _unpack(self, batch):
        """ _unpack """
        assert isinstance(batch, list), \
                'batch should be a list filled with tuples (img, label)'
        bs = len(batch)
        assert bs > 0, 'size of the batch data should > 0'
        #imgs, labels = list(zip(*batch))
        imgs = []
        labels = []
        for item in batch:
            imgs.append(item[0])
            labels.append(item[1])
        return np.array(imgs), np.array(labels), bs

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """Mixup and Cutmix operator"""

    def __init__(self,
                 mixup_alpha: float=1.,
                 cutmix_alpha: float=0.,
                 switch_prob: float=0.5):
        """Build Mixup operator

        Args:
            mixup_alpha (float, optional): The parameter alpha of mixup, mixup is active if > 0. Defaults to 1..
            cutmix_alpha (float, optional): The parameter alpha of cutmix, cutmix is active if > 0. Defaults to 0..
            switch_prob (float, optional): The probability of switching to cutmix instead of mixup when both are active. Defaults to 0.5.

        Raises:
            Exception: The value of parameters are illegal.
        """
        if mixup_alpha <= 0 and cutmix_alpha <= 0:
            raise Exception(
                f"At least one of parameter alpha of Mixup and Cutmix is greater than 0. mixup_alpha: {mixup_alpha}, cutmix_alpha: {cutmix_alpha}"
            )
        self._mixup_alpha = mixup_alpha
        self._cutmix_alpha = cutmix_alpha
        self._switch_prob = switch_prob

    def _mixup(self, imgs, labels, bs):
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._mixup_alpha, self._mixup_alpha)
        lams = np.array([lam] * bs, dtype=np.float32)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], lams))

    def _rand_bbox(self, size, lam):
        """ _rand_bbox """
        w = size[2]
        h = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        # uniform
        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def _cutmix(self, imgs, labels, bs):
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._cutmix_alpha, self._cutmix_alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        lams = np.array([lam] * bs, dtype=np.float32)
        return list(zip(imgs, labels, labels[idx], lams))

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        if np.random.rand() < self._switch_prob:
            return self._cutmix(imgs, labels, bs)
        else:
            return self._mixup(imgs, labels, bs)


class CutmixOperator(BatchOperator):
    def __init__(self, **kwargs):
        raise Exception(
            f"\"CutmixOperator\" has been deprecated. Please use MixupOperator with \"cutmix_alpha\" and \"switch_prob\" to enable Cutmix. Refor to doc for details."
        )


class FmixOperator(BatchOperator):
    """ Fmix operator """

    def __init__(self, alpha=1, decay_power=3, max_soft=0., reformulate=False):
        self._alpha = alpha
        self._decay_power = decay_power
        self._max_soft = max_soft
        self._reformulate = reformulate

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        size = (imgs.shape[2], imgs.shape[3])
        lam, mask = sample_mask(self._alpha, self._decay_power, \
                size, self._max_soft, self._reformulate)
        imgs = mask * imgs + (1 - mask) * imgs[idx]
        return list(zip(imgs, labels, labels[idx], [lam] * bs))
