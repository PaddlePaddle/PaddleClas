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
import random

import numpy as np

from ppcls.utils import logger
from ppcls.data.preprocess.ops.fmix import sample_mask

import paddle
import paddle.nn.functional as F


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

    def _one_hot(self, targets):
        return np.eye(self.class_num, dtype="float32")[targets]

    def _mix_target(self, targets0, targets1, lam):
        one_hots0 = self._one_hot(targets0)
        one_hots1 = self._one_hot(targets1)
        return one_hots0 * lam + one_hots1 * (1 - lam)

    def __call__(self, batch):
        return batch


class MixupOperator(BatchOperator):
    """ Mixup operator 
    reference: https://arxiv.org/abs/1710.09412

    """

    def __init__(self, class_num, alpha: float=1.):
        """Build Mixup operator

        Args:
            alpha (float, optional): The parameter alpha of mixup. Defaults to 1..

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Mixup should be greater than 0. \"alpha\": {alpha}."
            )
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"MixupOperator\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        self._alpha = alpha
        self.class_num = class_num

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)
        imgs = lam * imgs + (1 - lam) * imgs[idx]
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class CutmixOperator(BatchOperator):
    """ Cutmix operator
    reference: https://arxiv.org/abs/1905.04899

    """

    def __init__(self, class_num, alpha=0.2):
        """Build Cutmix operator

        Args:
            alpha (float, optional): The parameter alpha of cutmix. Defaults to 0.2.

        Raises:
            Exception: The value of parameter is illegal.
        """
        if alpha <= 0:
            raise Exception(
                f"Parameter \"alpha\" of Cutmix should be greater than 0. \"alpha\": {alpha}."
            )
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"CutmixOperator\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        self._alpha = alpha
        self.class_num = class_num

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

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        lam = np.random.beta(self._alpha, self._alpha)

        bbx1, bby1, bbx2, bby2 = self._rand_bbox(imgs.shape, lam)
        imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[idx, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - (float(bbx2 - bbx1) * (bby2 - bby1) /
                   (imgs.shape[-2] * imgs.shape[-1]))
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class FmixOperator(BatchOperator):
    """ Fmix operator 
    reference: https://arxiv.org/abs/2002.12047
    
    """

    def __init__(self,
                 class_num,
                 alpha=1,
                 decay_power=3,
                 max_soft=0.,
                 reformulate=False):
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"FmixOperator\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        self._alpha = alpha
        self._decay_power = decay_power
        self._max_soft = max_soft
        self._reformulate = reformulate
        self.class_num = class_num

    def __call__(self, batch):
        imgs, labels, bs = self._unpack(batch)
        idx = np.random.permutation(bs)
        size = (imgs.shape[2], imgs.shape[3])
        lam, mask = sample_mask(self._alpha, self._decay_power, \
                size, self._max_soft, self._reformulate)
        imgs = mask * imgs + (1 - mask) * imgs[idx]
        targets = self._mix_target(labels, labels[idx], lam)
        return list(zip(imgs, targets))


class OpSampler(object):
    """ Sample a operator from  """

    def __init__(self, class_num, **op_dict):
        """Build OpSampler

        Raises:
            Exception: The parameter \"prob\" of operator(s) are be set error.
        """
        if not class_num:
            msg = "Please set \"Arch.class_num\" in config if use \"OpSampler\"."
            logger.error(Exception(msg))
            raise Exception(msg)

        if len(op_dict) < 1:
            msg = f"ConfigWarning: No operator in \"OpSampler\". \"OpSampler\" has been skipped."
            logger.warning(msg)

        self.ops = {}
        total_prob = 0
        for op_name in op_dict:
            param = op_dict[op_name]
            if "prob" not in param:
                msg = f"ConfigWarning: Parameter \"prob\" should be set when use operator in \"OpSampler\". The operator \"{op_name}\"'s prob has been set \"0\"."
                logger.warning(msg)
            prob = param.pop("prob", 0)
            total_prob += prob
            param.update({"class_num": class_num})
            op = eval(op_name)(**param)
            self.ops.update({op: prob})

        if total_prob > 1:
            msg = f"ConfigError: The total prob of operators in \"OpSampler\" should be less 1."
            logger.error(Exception(msg))
            raise Exception(msg)

        # add "None Op" when total_prob < 1, "None Op" do nothing
        self.ops[None] = 1 - total_prob

    def __call__(self, batch):
        op = random.choices(
            list(self.ops.keys()), weights=list(self.ops.values()), k=1)[0]
        # return batch directly when None Op
        return op(batch) if op else batch


class MixupCutmixHybrid(object):
    """ Mixup/Cutmix that applies different params to each element or whole batch

    Args:
        mixup_alpha (float): mixup alpha value, mixup is active if > 0.
        cutmix_alpha (float): cutmix alpha value, cutmix is active if > 0.
        cutmix_minmax (List[float]): cutmix min/max image ratio, cutmix is active and uses this vs alpha if not None.
        prob (float): probability of applying mixup or cutmix per batch or element
        switch_prob (float): probability of switching to cutmix instead of mixup when both are active
        mode (str): how to apply mixup/cutmix params (per 'batch', 'pair' (pair of elements), 'elem' (element)
        correct_lam (bool): apply lambda correction when cutmix bbox clipped by image borders
        label_smoothing (float): apply label smoothing to the mixed target tensor
        num_classes (int): number of classes for target
    """

    def __init__(self,
                 mixup_alpha=1.,
                 cutmix_alpha=0.,
                 cutmix_minmax=None,
                 prob=1.0,
                 switch_prob=0.5,
                 mode='batch',
                 correct_lam=True,
                 label_smoothing=0.1,
                 num_classes=4):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_minmax = cutmix_minmax
        if self.cutmix_minmax is not None:
            assert len(self.cutmix_minmax) == 2
            # force cutmix alpha == 1.0 when minmax active to keep logic simple & safe
            self.cutmix_alpha = 1.0
        self.mix_prob = prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.mode = mode
        self.correct_lam = correct_lam  # correct lambda based on clipped area for cutmix
        self.mixup_enabled = True  # set to false to disable mixing (intended tp be set by train loop)

    def _one_hot(self, x, num_classes, on_value=1., off_value=0.):
        x = paddle.cast(x, dtype='int64')
        on_value = paddle.full([x.shape[0], num_classes], on_value)
        off_value = paddle.full([x.shape[0], num_classes], off_value)
        return paddle.where(
            F.one_hot(x, num_classes) == 1, on_value, off_value)

    def _mixup_target(self, target, num_classes, lam=1., smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        y1 = self._one_hot(
            target,
            num_classes,
            on_value=on_value,
            off_value=off_value, )
        y2 = self._one_hot(
            target.flip(0),
            num_classes,
            on_value=on_value,
            off_value=off_value)
        return y1 * lam + y2 * (1. - lam)

    def _rand_bbox(self, img_shape, lam, margin=0., count=None):
        """ Standard CutMix bounding-box
        Generates a random square bbox based on lambda value. This impl includes
        support for enforcing a border margin as percent of bbox dimensions.

        Args:
            img_shape (tuple): Image shape as tuple
            lam (float): Cutmix lambda value
            margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
            count (int): Number of bbox to generate
        """
        ratio = np.sqrt(1 - lam)
        img_h, img_w = img_shape[-2:]
        cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
        margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
        cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
        cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
        yl = np.clip(cy - cut_h // 2, 0, img_h)
        yh = np.clip(cy + cut_h // 2, 0, img_h)
        xl = np.clip(cx - cut_w // 2, 0, img_w)
        xh = np.clip(cx + cut_w // 2, 0, img_w)
        return yl, yh, xl, xh

    def _rand_bbox_minmax(self, img_shape, minmax, count=None):
        """ Min-Max CutMix bounding-box
        Inspired by Darknet cutmix impl, generates a random rectangular bbox
        based on min/max percent values applied to each dimension of the input image.

        Typical defaults for minmax are usually in the  .2-.3 for min and .8-.9 range for max.

        Args:
            img_shape (tuple): Image shape as tuple
            minmax (tuple or list): Min and max bbox ratios (as percent of image size)
            count (int): Number of bbox to generate
        """
        assert len(minmax) == 2
        img_h, img_w = img_shape[-2:]
        cut_h = np.random.randint(
            int(img_h * minmax[0]), int(img_h * minmax[1]), size=count)
        cut_w = np.random.randint(
            int(img_w * minmax[0]), int(img_w * minmax[1]), size=count)
        yl = np.random.randint(0, img_h - cut_h, size=count)
        xl = np.random.randint(0, img_w - cut_w, size=count)
        yu = yl + cut_h
        xu = xl + cut_w
        return yl, yu, xl, xu

    def _cutmix_bbox_and_lam(self,
                             img_shape,
                             lam,
                             ratio_minmax=None,
                             correct_lam=True,
                             count=None):
        """ Generate bbox and apply lambda correction.
        """
        if ratio_minmax is not None:
            yl, yu, xl, xu = self._rand_bbox_minmax(
                img_shape, ratio_minmax, count=count)
        else:
            yl, yu, xl, xu = self._rand_bbox(img_shape, lam, count=count)
        if correct_lam or ratio_minmax is not None:
            bbox_area = (yu - yl) * (xu - xl)
            lam = 1. - bbox_area / float(img_shape[-2] * img_shape[-1])
        return (yl, yu, xl, xu), lam

    def _params_per_elem(self, batch_size):
        lam = np.ones(batch_size, dtype=np.float32)
        use_cutmix = np.zeros(batch_size, dtype=np.bool)
        if self.mixup_enabled:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand(batch_size) < self.switch_prob
                lam_mix = np.where(
                    use_cutmix,
                    np.random.beta(
                        self.cutmix_alpha, self.cutmix_alpha, size=batch_size),
                    np.random.beta(
                        self.mixup_alpha, self.mixup_alpha, size=batch_size))
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(
                    self.mixup_alpha, self.mixup_alpha, size=batch_size)
            elif self.cutmix_alpha > 0.:
                use_cutmix = np.ones(batch_size, dtype=np.bool)
                lam_mix = np.random.beta(
                    self.cutmix_alpha, self.cutmix_alpha, size=batch_size)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = np.where(
                np.random.rand(batch_size) < self.mix_prob,
                lam_mix.astype(np.float32), lam)
        return lam, use_cutmix

    def _params_per_batch(self):
        lam = 1.
        use_cutmix = False
        if self.mixup_enabled and np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0. and self.cutmix_alpha > 0.:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha) if use_cutmix else \
                    np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.mixup_alpha > 0.:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            else:
                assert False, "One of mixup_alpha > 0., cutmix_alpha > 0., cutmix_minmax not None should be true."
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_elem(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size)
        x_orig = x.clone(
        )  # need to keep an unmodified original for mixing source
        for i in range(batch_size):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = self._cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam)
                    if yl < yh and xl < xh:
                        x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
        return paddle.to_tensor(lam_batch, dtype=x.dtype).unsqueeze(1)

    def _mix_pair(self, x):
        batch_size = len(x)
        lam_batch, use_cutmix = self._params_per_elem(batch_size // 2)
        x_orig = x.clone(
        )  # need to keep an unmodified original for mixing source
        for i in range(batch_size // 2):
            j = batch_size - i - 1
            lam = lam_batch[i]
            if lam != 1.:
                if use_cutmix[i]:
                    (yl, yh, xl, xh), lam = self._cutmix_bbox_and_lam(
                        x[i].shape,
                        lam,
                        ratio_minmax=self.cutmix_minmax,
                        correct_lam=self.correct_lam)
                    if yl < yh and xl < xh:
                        x[i][:, yl:yh, xl:xh] = x_orig[j][:, yl:yh, xl:xh]
                        x[j][:, yl:yh, xl:xh] = x_orig[i][:, yl:yh, xl:xh]
                    lam_batch[i] = lam
                else:
                    x[i] = x[i] * lam + x_orig[j] * (1 - lam)
                    x[j] = x[j] * lam + x_orig[i] * (1 - lam)
        lam_batch = np.concatenate((lam_batch, lam_batch[::-1]))
        return paddle.to_tensor(lam_batch, dtype=x.dtype).unsqueeze(1)

    def _mix_batch(self, x):
        lam, use_cutmix = self._params_per_batch()
        if lam == 1.:
            return 1.
        if use_cutmix:
            (yl, yh, xl, xh), lam = self._cutmix_bbox_and_lam(
                x.shape,
                lam,
                ratio_minmax=self.cutmix_minmax,
                correct_lam=self.correct_lam)
            if yl < yh and xl < xh:
                x[:, :, yl:yh, xl:xh] = x.flip(0)[:, :, yl:yh, xl:xh]

        else:
            x_flipped = x.flip(0) * (1. - lam)
            x[:] = x * lam + x_flipped
        return lam

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
        x, target, bs = self._unpack(batch)
        x = paddle.to_tensor(x)
        target = paddle.to_tensor(target)
        assert len(x) % 2 == 0, 'Batch size should be even when using this'
        if self.mode == 'elem':
            lam = self._mix_elem(x)
        elif self.mode == 'pair':
            lam = self._mix_pair(x)
        else:
            lam = self._mix_batch(x)
        target = self._mixup_target(target, self.num_classes, lam,
                                    self.label_smoothing)

        return list(zip(x.numpy(), target.numpy()))
