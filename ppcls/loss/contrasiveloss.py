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

from typing import Dict

import paddle
import paddle.nn as nn
from ppcls.loss.xbm import CrossBatchMemory


class ContrastiveLoss(nn.Layer):
    """ContrastiveLoss

    Args:
        margin (float): margin
        feat_dim (int): feature dim
        feature_from (str, optional): key which features fetched from output dict. Defaults to "features".
    """

    def __init__(self,
                 margin: float,
                 feat_dim: int,
                 normalize_feature=True,
                 epsilon: float = 1e-5,
                 feature_from: str = "features"):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.feat_dim = feat_dim
        self.epsilon = epsilon
        self.feature_from = feature_from

    def __call__(self, input: Dict[str, paddle.Tensor], target: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        feats = input[self.feature_from]
        labels = target

        # normalize along feature dim
        if self.normalize_feature:
            feats = nn.functional.normalize(feats, p=2, axis=1)

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        loss = self._compute_loss(feats, target, feats, target)

        return {'ContrastiveLoss': loss}

    def _compute_loss(self,
                      inputs_q: paddle.Tensor, targets_q: paddle.Tensor,
                      inputs_k: paddle.Tensor, targets_k: paddle.Tensor) -> paddle.Tensor:
        batch_size = inputs_q.shape[0]
        # Compute similarity matrix
        sim_mat = paddle.matmul(inputs_q, inputs_k.t())
        loss = []

        # neg_count = []
        for i in range(batch_size):
            pos_pair_ = paddle.masked_select(sim_mat[i], targets_q[i] == targets_k)
            pos_pair_ = paddle.masked_select(pos_pair_, pos_pair_ < 1 - self.epsilon)

            neg_pair_ = paddle.masked_select(sim_mat[i], targets_q[i] != targets_k)
            neg_pair = paddle.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = paddle.sum(-pos_pair_ + 1)

            if len(neg_pair) > 0:
                neg_loss = paddle.sum(neg_pair)
                # neg_count.append(len(neg_pair))
            else:
                neg_loss = 0
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / batch_size  # / all_targets.shape[1]
        return loss


class ContrastiveLoss_XBM(nn.Layer):
    """ContrastiveLoss_XBM

    Args:
        margin (float): margin
        feat_dim (int): feature dim
        feature_from (str, optional): key which features fetched from output dict. Defaults to "features".
    """

    def __init__(self,
                 margin: float,
                 feat_dim: int,
                 start_iter: int,
                 xbm_size: int,
                 xbm_weight: int,
                 normalize_feature=True,
                 epsilon: float = 1e-5,
                 feature_from: str = "features"):
        super(ContrastiveLoss_XBM, self).__init__()
        self.margin = margin
        self.feat_dim = feat_dim
        self.start_iter = start_iter
        self.epsilon = epsilon
        self.feature_from = feature_from
        self.normalize_feature = normalize_feature
        self.iter = 0
        self.xbm = CrossBatchMemory(xbm_size, feat_dim)
        self.xbm_weight = xbm_weight

    def __call__(self, input: Dict[str, paddle.Tensor], target: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        feats = input[self.feature_from]
        labels = target

        # normalize along feature dim
        if self.normalize_feature:
            feats = nn.functional.normalize(feats, p=2, axis=1)

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        loss = self._compute_loss(feats, labels, feats, labels)

        # for XBM loss
        # if not hasattr(self, 'model'):
        #     raise ValueError("model should assign to ContrastiveLoss_XBM")
        self.iter += 1
        if self.iter > self.start_iter:
            self.xbm.enqueue_dequeue(feats.detach(), labels.detach())
            xbm_feats, xbm_labels = self.xbm.get()
            xbm_loss = self._compute_loss(feats, labels, xbm_feats, xbm_labels)
            loss = loss + self.xbm_weight * xbm_loss

        return {'ContrastiveLoss_XBM': loss}

    def _compute_loss(self,
                      inputs_q: paddle.Tensor, targets_q: paddle.Tensor,
                      inputs_k: paddle.Tensor, targets_k: paddle.Tensor) -> paddle.Tensor:
        batch_size = inputs_q.shape[0]
        # Compute similarity matrix
        sim_mat = paddle.matmul(inputs_q, inputs_k.t())
        loss = []

        # neg_count = []
        for i in range(batch_size):
            pos_pair_ = paddle.masked_select(sim_mat[i], targets_q[i] == targets_k)
            pos_pair_ = paddle.masked_select(pos_pair_, pos_pair_ < 1 - self.epsilon)

            neg_pair_ = paddle.masked_select(sim_mat[i], targets_q[i] != targets_k)
            neg_pair = paddle.masked_select(neg_pair_, neg_pair_ > self.margin)

            pos_loss = paddle.sum(-pos_pair_ + 1)

            if len(neg_pair) > 0:
                neg_loss = paddle.sum(neg_pair)
                # neg_count.append(len(neg_pair))
            else:
                neg_loss = 0
            loss.append(pos_loss + neg_loss)

        loss = sum(loss) / batch_size  # / all_targets.shape[1]
        return loss
