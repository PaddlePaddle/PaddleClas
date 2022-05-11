from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def ratio2weight(targets, ratio):
    #     print(targets)
    pos_weights = targets * (1. - ratio)
    neg_weights = (1. - targets) * ratio
    weights = paddle.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1)

    return weights


class BCELoss(nn.Layer):
    """BCE Loss.
    Args:
        
    """

    def __init__(self,
                 sample_weight=True,
                 size_sum=True,
                 smoothing=None,
                 weight=1.0):
        super(BCELoss, self).__init__()

        self.sample_weight = sample_weight
        self.size_sum = size_sum
        self.hyper = 0.8
        self.smoothing = smoothing

    def forward(self, logits, labels):
        targets, ratio = labels

        if self.smoothing is not None:
            targets = (1 - self.smoothing) * targets + self.smoothing * (
                1 - targets)

        targets = paddle.cast(targets, 'float32')

        loss_m = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none')

        targets_mask = paddle.cast(targets > 0.5, 'float32')
        if self.sample_weight:
            weight = ratio2weight(targets_mask, ratio[0])
            weight = weight * (targets > -1)
            loss_m = loss_m * weight

        loss = loss_m.sum(1).mean() if self.size_sum else loss_m.sum()

        return {"BCELoss": loss}
