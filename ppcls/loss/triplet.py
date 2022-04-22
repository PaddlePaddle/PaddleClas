from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from typing import Tuple

import paddle
import paddle.nn as nn


class TripletLossV2(nn.Layer):
    """Triplet loss with hard positive/negative mining.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=0.5, normalize_feature=True):
        super(TripletLossV2, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)
        self.normalize_feature = normalize_feature

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        inputs = input["backbone"]

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
        return {"TripletLossV2": loss}


class TripletLoss(nn.Layer):
    """Triplet loss with hard positive/negative mining.
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    Args:
        margin (float): margin for triplet.
    """

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(margin=margin)

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (num_classes)
        """
        inputs = input["features"]

        bs = inputs.shape[0]
        # Compute pairwise distance, replace by the official when merged
        dist = paddle.pow(inputs, 2).sum(axis=1, keepdim=True).expand([bs, bs])
        dist = dist + dist.t()
        dist = paddle.addmm(
            input=dist, x=inputs, y=inputs.t(), alpha=-2.0, beta=1.0)
        dist = paddle.clip(dist, min=1e-12).sqrt()

        mask = paddle.equal(
            target.expand([bs, bs]), target.expand([bs, bs]).t())
        mask_numpy_idx = mask.numpy()
        dist_ap, dist_an = [], []
        for i in range(bs):
            # dist_ap_i = paddle.to_tensor(dist[i].numpy()[mask_numpy_idx[i]].max(),dtype='float64').unsqueeze(0)
            # dist_ap_i.stop_gradient = False
            # dist_ap.append(dist_ap_i)
            dist_ap.append(
                max([
                    dist[i][j] if mask_numpy_idx[i][j] == True else float(
                        "-inf") for j in range(bs)
                ]).unsqueeze(0))
            # dist_an_i = paddle.to_tensor(dist[i].numpy()[mask_numpy_idx[i] == False].min(), dtype='float64').unsqueeze(0)
            # dist_an_i.stop_gradient = False
            # dist_an.append(dist_an_i)
            dist_an.append(
                min([
                    dist[i][k] if mask_numpy_idx[i][k] == False else float(
                        "inf") for k in range(bs)
                ]).unsqueeze(0))

        dist_ap = paddle.concat(dist_ap, axis=0)
        dist_an = paddle.concat(dist_an, axis=0)

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return {"TripletLoss": loss}


class TripletLossV3(nn.Layer):
    """Modified from Tong Xiao's open-reid (https://github.com/Cysu/open-reid).
    Related Triplet Loss theory can be found in paper 'In Defense of the Triplet
    Loss for Person Re-Identification'."""

    def __init__(self, margin=None, normalize_feature=False):
        super(TripletLossV3, self).__init__()
        self.normalize_feature = normalize_feature
        self.margin = margin
        if margin is not None:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, input, target):
        global_feat = input["backbone"]
        if self.normalize_feature:
            global_feat = self._normalize(global_feat, axis=-1)
        dist_mat = self._euclidean_dist(global_feat, global_feat)
        dist_ap, dist_an = self._hard_example_mining(dist_mat, target)
        y = paddle.ones_like(dist_an)
        if self.margin is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)

        return {"TripletLossV3": loss}

    def _normalize(self, x: paddle.Tensor, axis: int=-1) -> paddle.Tensor:
        """Normalizing to unit length along the specified dimension.

        Args:
            x (paddle.Tensor): (batch_size, feature_dim)
            axis (int, optional): normalization dim. Defaults to -1.

        Returns:
            paddle.Tensor: (batch_size, feature_dim)
        """
        x = 1. * x / (paddle.norm(
            x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
        return x

    def _euclidean_dist(self, x: paddle.Tensor,
                        y: paddle.Tensor) -> paddle.Tensor:
        """compute euclidean distance between two batched vectors

        Args:
            x (paddle.Tensor): (N, feature_dim)
            y (paddle.Tensor): (M, feature_dim)

        Returns:
            paddle.Tensor: (N, M)
        """
        m, n = x.shape[0], y.shape[0]
        d = x.shape[1]
        xx = paddle.pow(x, 2).sum(1, keepdim=True).expand([m, n])
        yy = paddle.pow(y, 2).sum(1, keepdim=True).expand([n, m]).t()
        dist = xx + yy
        dist = dist.addmm(x, y.t(), alpha=-2, beta=1)
        # dist = dist - 2*(x@y.t())
        dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
        return dist

    def _hard_example_mining(
            self,
            dist_mat: paddle.Tensor,
            labels: paddle.Tensor,
            return_inds: bool=False) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """For each anchor, find the hardest positive and negative sample.

        Args:
            dist_mat (paddle.Tensor): pair wise distance between samples, [N, N]
            labels (paddle.Tensor): labels, [N, ]
            return_inds (bool, optional): whether to return the indices . Defaults to False.

        Returns:
            Tuple[paddle.Tensor, paddle.Tensor]: [(N, ), (N, )]

        NOTE: Only consider the case in which all labels have same num of samples,
        thus we can cope with all anchors in parallel.
        """
        assert len(dist_mat.shape) == 2
        assert dist_mat.shape[0] == dist_mat.shape[1]
        N = dist_mat.shape[0]

        # shape [N, N]
        is_pos = labels.expand([N, N]).equal(labels.expand([N, N]).t())
        is_neg = labels.expand([N, N]).not_equal(labels.expand([N, N]).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = paddle.max(dist_mat[is_pos].reshape([N, -1]),
                             1,
                             keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an = paddle.min(dist_mat[is_neg].reshape([N, -1]),
                             1,
                             keepdim=True)
        # shape [N]
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        if return_inds:
            # shape [N, N]
            ind = (labels.new().resize_as_(labels)
                   .copy_(paddle.arange(0, N).long())
                   .unsqueeze(0).expand(N, N))
            # shape [N, 1]
            p_inds = paddle.gather(ind[is_pos].reshape([N, -1]), 1,
                                   relative_p_inds.data)
            n_inds = paddle.gather(ind[is_neg].reshape([N, -1]), 1,
                                   relative_n_inds.data)
            # shape [N]
            p_inds = p_inds.squeeze(1)
            n_inds = n_inds.squeeze(1)
            return dist_ap, dist_an, p_inds, n_inds

        return dist_ap, dist_an
