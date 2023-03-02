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

# reference: https://arxiv.org/abs/2011.14670

import copy
import numpy as np
import paddle
from paddle import nn
from paddle.nn import functional as F

from .dist_loss import cosine_similarity
from .celoss import CELoss


def euclidean_dist(x, y):
    m, n = x.shape[0], y.shape[0]
    xx = paddle.pow(x, 2).sum(1, keepdim=True).expand([m, n])
    yy = paddle.pow(y, 2).sum(1, keepdim=True).expand([n, m]).t()
    dist = xx + yy - 2 * paddle.matmul(x, y.t())
    dist = dist.clip(min=1e-12).sqrt()  # for numerical stability
    return dist


def hard_example_mining(dist_mat, is_pos, is_neg):
    """For each anchor, find the hardest positive and negative sample.
    Args:
        dist_mat: pairwise distance between samples, shape [N, M]
        is_pos: positive index with shape [N, M]
        is_neg: negative index with shape [N, M]
    Returns:
        dist_ap: distance(anchor, positive); shape [N, 1]
        dist_an: distance(anchor, negative); shape [N, 1]
    """

    inf = float("inf")

    def _masked_max(tensor, mask, axis):
        masked = paddle.multiply(tensor, mask.astype(tensor.dtype))
        neg_inf = paddle.zeros_like(tensor)
        neg_inf.stop_gradient = True
        neg_inf[paddle.logical_not(mask)] = -inf
        return paddle.max(masked + neg_inf, axis=axis, keepdim=True)

    def _masked_min(tensor, mask, axis):
        masked = paddle.multiply(tensor, mask.astype(tensor.dtype))
        pos_inf = paddle.zeros_like(tensor)
        pos_inf.stop_gradient = True
        pos_inf[paddle.logical_not(mask)] = inf
        return paddle.min(masked + pos_inf, axis=axis, keepdim=True)

    assert len(dist_mat.shape) == 2
    dist_ap = _masked_max(dist_mat, is_pos, axis=1)
    dist_an = _masked_min(dist_mat, is_neg, axis=1)
    return dist_ap, dist_an


class IntraDomainScatterLoss(nn.Layer):
    """
    IntraDomainScatterLoss
    
    enhance intra-domain diversity and disarrange inter-domain distributions like confusing multiple styles.

    reference: https://arxiv.org/abs/2011.14670
    """

    def __init__(self, normalize_feature, feature_from):
        super(IntraDomainScatterLoss, self).__init__()
        self.normalize_feature = normalize_feature
        self.feature_from = feature_from

    def forward(self, input, batch):
        domains = batch["domain"]
        inputs = input[self.feature_from]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        unique_label = paddle.unique(domains)
        features_per_domain = list()
        for i, x in enumerate(unique_label):
            features_per_domain.append(inputs[x == domains])
        num_domain = len(features_per_domain)
        losses = []
        for i in range(num_domain):
            features_in_same_domain = features_per_domain[i]
            center = paddle.mean(features_in_same_domain, 0)
            cos_sim = cosine_similarity(
                center.unsqueeze(0), features_in_same_domain)
            losses.append(paddle.mean(cos_sim))
        loss = paddle.mean(paddle.stack(losses))
        return {"IntraDomainScatterLoss": loss}


class InterDomainShuffleLoss(nn.Layer):
    """
    InterDomainShuffleLoss

    pull the negative sample of the interdomain and push the negative sample of the intra-domain, 
    so that the inter-domain distributions are shuffled.

    reference: https://arxiv.org/abs/2011.14670
    """

    def __init__(self, normalize_feature=True, feature_from="features"):
        super(InterDomainShuffleLoss, self).__init__()
        self.feature_from = feature_from
        self.normalize_feature = normalize_feature

    def forward(self, input, batch):
        target = batch["label"]
        domains = batch["domain"]
        inputs = input[self.feature_from]
        bs = inputs.shape[0]

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        # compute distance
        dist_mat = euclidean_dist(inputs, inputs)

        is_same_img = np.zeros(shape=[bs, bs], dtype=bool)
        np.fill_diagonal(is_same_img, True)
        is_same_img = paddle.to_tensor(is_same_img)
        is_diff_instance = target.reshape([bs, 1]).expand([bs, bs])\
            .not_equal(target.reshape([bs, 1]).expand([bs, bs]).t())
        is_same_domain = domains.reshape([bs, 1]).expand([bs, bs])\
            .equal(domains.reshape([bs, 1]).expand([bs, bs]).t())
        is_diff_domain = is_same_domain == False

        is_pos = paddle.logical_or(is_same_img, is_diff_domain)
        is_neg = paddle.logical_and(is_diff_instance, is_same_domain)

        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)

        y = paddle.ones_like(dist_an)
        loss = F.soft_margin_loss(dist_an - dist_ap, y)
        if loss == float('Inf'):
            loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=0.3)
        return {"InterDomainShuffleLoss": loss}


class CELossForMetaBIN(CELoss):
    def _labelsmoothing(self, target, class_num):
        if len(target.shape) == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        # epsilon is different from the one in original CELoss
        epsilon = class_num / (class_num - 1) * self.epsilon
        soft_target = F.label_smooth(one_hot_target, epsilon=epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def forward(self, x, batch):
        label = batch["label"]
        return super().forward(x, label)


class TripletLossForMetaBIN(nn.Layer):
    def __init__(self,
                 margin=1,
                 normalize_feature=False,
                 feature_from="feature"):
        super(TripletLossForMetaBIN, self).__init__()
        self.margin = margin
        self.feature_from = feature_from
        self.normalize_feature = normalize_feature

    def forward(self, input, batch):
        inputs = input[self.feature_from]
        targets = batch["label"]
        bs = inputs.shape[0]
        all_targets = targets

        if self.normalize_feature:
            inputs = 1. * inputs / (paddle.expand_as(
                paddle.norm(
                    inputs, p=2, axis=-1, keepdim=True), inputs) + 1e-12)

        dist_mat = euclidean_dist(inputs, inputs)

        is_pos = all_targets.reshape([bs, 1]).expand([bs, bs]).equal(
            all_targets.reshape([bs, 1]).expand([bs, bs]).t())
        is_neg = all_targets.reshape([bs, 1]).expand([bs, bs]).not_equal(
            all_targets.reshape([bs, 1]).expand([bs, bs]).t())
        dist_ap, dist_an = hard_example_mining(dist_mat, is_pos, is_neg)

        y = paddle.ones_like(dist_an)
        loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=self.margin)
        return {"TripletLoss": loss}
