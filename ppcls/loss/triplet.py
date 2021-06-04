from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn


class TripletLoss(nn.Layer):
    """Triplet loss with hard positive/negative mining.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.5, normalize_feature=True):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        inputs = input["features"]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        bs = inputs.shape[0]

        # compute distance
        dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand([bs, bs])
        dist = dist + dist.t()
        dist = paddle.addmm(
            input=dist, x=inputs, y=inputs.t(), alpha=-2.0, beta=1.0)
        dist = paddle.clip(dist, min=1e-12).sqrt()

        # hard negative mining
        is_pos = paddle.expand(target, (
            bs, bs)).equal(paddle.expand(target, (bs, bs)).t())
        is_neg = paddle.expand(target, (
            bs, bs)).not_equal(paddle.expand(target, (bs, bs)).t())

        # `dist_ap` means distance(anchor, positive)
        ## both `dist_ap` and `relative_p_inds` with shape [N, 1]
        '''
        dist_ap, relative_p_inds = paddle.max(
            paddle.reshape(dist[is_pos], (bs, -1)), axis=1, keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an, relative_n_inds = paddle.min(
            paddle.reshape(dist[is_neg], (bs, -1)), axis=1, keepdim=True)
        '''
        dist_ap = paddle.max(paddle.reshape(
            paddle.masked_select(dist, is_pos), (bs, -1)),
                             axis=1,
                             keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an = paddle.min(paddle.reshape(
            paddle.masked_select(dist, is_neg), (bs, -1)),
                             axis=1,
                             keepdim=True)
        # shape [N]
        dist_ap = paddle.squeeze(dist_ap, axis=1)
        dist_an = paddle.squeeze(dist_an, axis=1)

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return {"TripletLoss": loss}
