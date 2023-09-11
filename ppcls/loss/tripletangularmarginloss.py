#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
from ppcls.loss.xbm import CrossBatchMemory


class TripletAngularMarginLoss(nn.Layer):
    """A more robust triplet loss with hard positive/negative mining on angular margin instead of relative distance between d(a,p) and d(a,n).

    Args:
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        reduction (str, optional): reducing option within an batch . Defaults to "mean".
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to False.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.9.
        an_value (float, optional): weight for d(a, n). Defaults to 0.5.
        feature_from (str, optional): which key feature from. Defaults to "features".
    """

    def __init__(self,
                 margin=0.5,
                 normalize_feature=True,
                 reduction="mean",
                 add_absolute=False,
                 absolute_loss_weight=1.0,
                 ap_value=0.9,
                 an_value=0.5,
                 feature_from="features"):
        super(TripletAngularMarginLoss, self).__init__()
        self.margin = margin
        self.feature_from = feature_from
        self.ranking_loss = paddle.nn.loss.MarginRankingLoss(
            margin=margin, reduction=reduction)
        self.normalize_feature = normalize_feature
        self.add_absolute = add_absolute
        self.ap_value = ap_value
        self.an_value = an_value
        self.absolute_loss_weight = absolute_loss_weight

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (batch_size)
        """
        inputs = input[self.feature_from]

        if self.normalize_feature:
            inputs = paddle.divide(
                inputs, paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True))

        bs = inputs.shape[0]

        # compute distance(cos-similarity)
        dist = paddle.matmul(inputs, inputs.t())

        # hard negative mining
        is_pos = paddle.expand(target, (
            bs, bs)).equal(paddle.expand(target, (bs, bs)).t())
        is_neg = paddle.expand(target, (
            bs, bs)).not_equal(paddle.expand(target, (bs, bs)).t())

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = paddle.min(paddle.reshape(
            paddle.masked_select(dist, is_pos), (bs, -1)),
                             axis=1,
                             keepdim=True)
        # `dist_an` means distance(anchor, negative)
        # both `dist_an` and `relative_n_inds` with shape [N, 1]
        dist_an = paddle.max(paddle.reshape(
            paddle.masked_select(dist, is_neg), (bs, -1)),
                             axis=1,
                             keepdim=True)
        # shape [N]
        dist_ap = paddle.squeeze(dist_ap, axis=1)
        dist_an = paddle.squeeze(dist_an, axis=1)

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)

        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = paddle.where(absolut_loss_ap > 0,
                                           absolut_loss_ap,
                                           paddle.zeros_like(absolut_loss_ap))

            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = paddle.where(absolut_loss_an > 0,
                                           absolut_loss_an,
                                           paddle.ones_like(absolut_loss_an))

            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()
                    ) * self.absolute_loss_weight + loss.mean()

        return {"TripletAngularMarginLoss": loss}


class TripletAngularMarginLoss_XBM(TripletAngularMarginLoss):
    """TripletAngularMarginLoss combined with CrossBatchMemory

    Args:
        start_iter: (int): from which step CrossBatchMemory is enabled
        xbm_size: (int): Size of CrossBatchMemory
        xbm_weight: (float): Weight of CrossBatchMemory loss
        feat_dim: (int): Channels of features in CrossBatchMemory
        margin (float, optional): angular margin. Defaults to 0.5.
        normalize_feature (bool, optional): whether to apply L2-norm in feature before computing distance(cos-similarity). Defaults to True.
        reduction (str, optional): reducing option within an batch . Defaults to "mean".
        add_absolute (bool, optional): whether add absolute loss within d(a,p) or d(a,n). Defaults to False.
        absolute_loss_weight (float, optional): weight for absolute loss. Defaults to 1.0.
        ap_value (float, optional): weight for d(a, p). Defaults to 0.9.
        an_value (float, optional): weight for d(a, n). Defaults to 0.5.
        feature_from (str, optional): which key feature from. Defaults to "features".
    """

    def __init__(self,
                 start_iter: int,
                 xbm_size: int,
                 xbm_weight: float,
                 feat_dim: int,
                 margin=0.5,
                 normalize_feature=True,
                 reduction="mean",
                 add_absolute=False,
                 absolute_loss_weight=1.0,
                 ap_value=0.9,
                 an_value=0.5,
                 feature_from="features"):
        super(TripletAngularMarginLoss_XBM, self).__init__(
            margin, normalize_feature, reduction, add_absolute,
            absolute_loss_weight, ap_value, an_value, feature_from)
        self.start_iter = start_iter
        self.xbm = CrossBatchMemory(xbm_size, feat_dim)
        self.xbm_weight = xbm_weight
        self.inf = 10  # 10 is big enough as inf for cos-similarity
        self.register_buffer("iter", paddle.to_tensor(0, dtype="int64"))

    def forward(self, input, target):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            target: ground truth labels with shape (batch_size)
        """
        feats = input[self.feature_from]
        if self.normalize_feature:
            feats = nn.functional.normalize(feats, p=2, axis=1)

        labels = target
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        loss = self._compute_loss(feats, labels, feats, labels)

        # XBM loss below
        self.iter += 1
        if self.iter.item() > self.start_iter:
            self.xbm.enqueue_dequeue(feats.detach(), labels.detach())
            xbm_feats, xbm_labels = self.xbm.get()
            xbm_loss = self._compute_loss(feats, labels, xbm_feats, xbm_labels)
            loss = loss + self.xbm_weight * xbm_loss

        return {"TripletAngularMarginLoss_XBM": loss}

    def _masked_max(self, tensor, mask, axis):
        masked = paddle.multiply(tensor, mask.astype(tensor.dtype))
        neg_inf = paddle.zeros_like(tensor)
        neg_inf.stop_gradient = True
        neg_inf[paddle.logical_not(mask)] = -self.inf
        return paddle.max(masked + neg_inf, axis=axis, keepdim=True)

    def _masked_min(self, tensor, mask, axis):
        masked = paddle.multiply(tensor, mask.astype(tensor.dtype))
        pos_inf = paddle.zeros_like(tensor)
        pos_inf.stop_gradient = True
        pos_inf[paddle.logical_not(mask)] = self.inf
        return paddle.min(masked + pos_inf, axis=axis, keepdim=True)

    def _compute_loss(self,
                      inputs_q: paddle.Tensor,
                      targets_q: paddle.Tensor,
                      inputs_k: paddle.Tensor,
                      targets_k: paddle.Tensor) -> paddle.Tensor:
        Q = inputs_q.shape[0]
        K = inputs_k.shape[0]

        # compute distance(cos-similarity)
        dist = paddle.matmul(inputs_q, inputs_k.t())  # [Q, K]

        # hard negative mining
        is_pos = paddle.expand(paddle.unsqueeze(targets_q, 1), (Q, K)).equal(
            paddle.expand(paddle.unsqueeze(targets_k, 1),
                          (K, Q)).t())  # [Q, K]
        is_neg = paddle.expand(paddle.unsqueeze(targets_q, 1),
                               (Q, K)).not_equal(
                                   paddle.expand(
                                       paddle.unsqueeze(targets_k, 1),
                                       (K, Q)).t())  # [Q, K]

        dist_ap = self._masked_min(dist, is_pos, axis=1)  # [Q, ]
        dist_an = self._masked_max(dist, is_neg, axis=1)  # [Q, ]

        # Compute ranking hinge loss
        y = paddle.ones_like(dist_an)
        loss = self.ranking_loss(dist_ap, dist_an, y)

        if self.add_absolute:
            absolut_loss_ap = self.ap_value - dist_ap
            absolut_loss_ap = paddle.where(absolut_loss_ap > 0,
                                           absolut_loss_ap,
                                           paddle.zeros_like(absolut_loss_ap))

            absolut_loss_an = dist_an - self.an_value
            absolut_loss_an = paddle.where(absolut_loss_an > 0,
                                           absolut_loss_an,
                                           paddle.ones_like(absolut_loss_an))

            loss = (absolut_loss_an.mean() + absolut_loss_ap.mean()
                    ) * self.absolute_loss_weight + loss.mean()

        return loss
