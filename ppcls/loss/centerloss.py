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


class CenterLoss(nn.Layer):
    """Center loss class

    Args:
        num_classes (int): number of classes.
        feat_dim (int): number of feature dimensions.
    """

    def __init__(self, num_classes: int, feat_dim: int):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        random_init_centers = paddle.randn(
            shape=[self.num_classes, self.feat_dim])
        self.centers = self.create_parameter(
            shape=(self.num_classes, self.feat_dim),
            default_initializer=nn.initializer.Assign(random_init_centers))
        self.add_parameter("centers", self.centers)

    def __call__(self, input: Dict[str, paddle.Tensor],
                 target: paddle.Tensor) -> Dict[str, paddle.Tensor]:
        """compute center loss.

        Args:
            input (Dict[str, paddle.Tensor]): {'features': (batch_size, feature_dim), ...}.
            target (paddle.Tensor): ground truth label with shape (batch_size, ).

        Returns:
            Dict[str, paddle.Tensor]: {'CenterLoss': loss}.
        """
        feats = input['backbone']
        labels = target

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        batch_size = feats.shape[0]
        distmat = paddle.pow(feats, 2).sum(axis=1, keepdim=True).expand([batch_size, self.num_classes]) + \
            paddle.pow(self.centers, 2).sum(axis=1, keepdim=True).expand([self.num_classes, batch_size]).t()
        distmat = distmat.addmm(x=feats, y=self.centers.t(), beta=1, alpha=-2)

        classes = paddle.arange(self.num_classes).astype(labels.dtype)
        labels = labels.unsqueeze(1).expand([batch_size, self.num_classes])
        mask = labels.equal(classes.expand([batch_size, self.num_classes]))

        dist = distmat * mask.astype(feats.dtype)
        loss = dist.clip(min=1e-12, max=1e+12).sum() / batch_size
        # return loss
        return {'CenterLoss': loss}
