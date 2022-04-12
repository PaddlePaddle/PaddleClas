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
from paddle import Tensor


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

    def __call__(self, input: Dict[str, Tensor],
                 target: Tensor) -> Dict[str, Tensor]:
        """compute center loss.

        Args:
            input (Dict[str, Tensor]): {'features': (batch_size, feature_dim), ...}.
            target (Tensor): ground truth label with shape (batch_size, ).

        Returns:
            Dict[str, Tensor]: {'CenterLoss': loss}.
        """
        feats = input['backbone']
        labels = target

        # squeeze labels to shape (batch_size, )
        if labels.ndim >= 2 and labels.shape[-1] == 1:
            labels = paddle.squeeze(labels, axis=[-1])

        batch_size = feats.shape[0]
        # calc feat * feat
        dist1 = paddle.sum(paddle.square(feats), axis=1, keepdim=True)
        dist1 = paddle.expand(dist1, [batch_size, self.num_classes])

        # dist2 of centers
        dist2 = paddle.sum(paddle.square(self.centers), axis=1,
                           keepdim=True)  # num_classes
        dist2 = paddle.expand(dist2, [self.num_classes, batch_size])
        dist2 = paddle.transpose(dist2, [1, 0])

        # first x * x + y * y
        distmat = paddle.add(dist1, dist2)

        tmp = paddle.matmul(feats, paddle.transpose(self.centers, [1, 0]))
        distmat = distmat - 2.0 * tmp

        # generate the mask
        classes = paddle.arange(self.num_classes)
        labels = paddle.expand(
            paddle.unsqueeze(labels, 1), (batch_size, self.num_classes))
        mask = paddle.equal(
            paddle.expand(classes, [batch_size, self.num_classes]),
            labels).astype("float32")  # get mask

        dist = paddle.multiply(distmat, mask)
        loss = paddle.sum(paddle.clip(dist, min=1e-12, max=1e+12)) / batch_size
        # return loss
        return {'CenterLoss': loss}
