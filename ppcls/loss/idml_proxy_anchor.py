# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is mainly based on https://github.com/wangck20/IDML/blob/main/image_retrieval/code/losses.py
# Paper: Introspective Deep Metric Learning

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import sklearn.preprocessing
from ppcls.utils.initializer import kaiming_normal_


def binarize(T, nb_classes):
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(T, classes=range(0, nb_classes))
    T = paddle.to_tensor(T)
    return T


def l2_norm(input, axis=1):
    norm = paddle.sqrt(
        paddle.sum(paddle.square(input), axis=axis, keepdim=True).add(paddle.to_tensor(1e-12))) 
    output = paddle.divide(input, norm)

    return output


class IDMLProxyAnchorLoss(nn.Layer):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, lr=100, feature_from="features"):
        paddle.nn.Layer.__init__(self)
        # Proxy Anchor Initialization
        
        kernel_weight = paddle.randn([nb_classes, sz_embed])
        kernel_weight = kaiming_normal_(kernel_weight, mode='fan_out')
        attr = paddle.ParamAttr(initializer=paddle.nn.initializer.Assign(kernel_weight),
                            learning_rate=lr)
        self.proxies = self.create_parameter(
            [nb_classes, sz_embed], attr=attr)

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.feature_from = feature_from

    def forward(self, inputs, label):
        inputs = inputs[self.feature_from]
        assert inputs.shape[0] == label.shape[0] * 2
        X, V = paddle.split(inputs, num_or_sections=2, axis=0)
        P = self.proxies
        X_mu = X
        X_var = V
        P_mu = P
        T = label
        P_var = paddle.zeros_like(P_mu)
        cos_ori = F.linear(l2_norm(X_mu), l2_norm(P_mu).t())
        sq_X = paddle.sum(paddle.pow(X_mu, 2), 1,
                          keepdim=True).expand([X_mu.shape[0], P_mu.shape[0]])
        sq_P = paddle.sum(paddle.pow(P_mu, 2),
                          1).expand([X_mu.shape[0], P_mu.shape[0]])
        distance_mu = sq_X + sq_P - 2 * paddle.mm(X_mu, P_mu.t())
        sq_X_var = paddle.sum(paddle.pow(X_var, 2), 1,
                              keepdim=True).expand([X_mu.shape[0],
                                                   P_mu.shape[0]])
        sq_P_var = paddle.sum(paddle.pow(P_var, 2),
                              1).expand([X_mu.shape[0],
                                        P_mu.shape[0]])
        distance_var = sq_X_var + sq_P_var - 2 * paddle.mm(X_var, P_var.t())
        cos = 1 - (1 - cos_ori) * paddle.exp(-paddle.divide(distance_var + 4, 5 * distance_mu))

        #cos = F.linear(l2_norm(X), l2_norm(P))  # Calcluate cosine similarity
        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        pos_exp = paddle.exp(-self.alpha * (cos - self.mrg))
        neg_exp = paddle.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = paddle.nonzero(P_one_hot.sum(axis=0) != 0).squeeze(
            axis=1)  # The set of positive proxies of data in the batch
        num_valid_proxies = len(
            with_pos_proxies)  # The number of positive proxies

        P_sim_sum = paddle.where(P_one_hot == 1, pos_exp,
                                 paddle.zeros_like(pos_exp)).sum(axis=0)
        N_sim_sum = paddle.where(N_one_hot == 1, neg_exp,
                                 paddle.zeros_like(neg_exp)).sum(axis=0)

        pos_term = paddle.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = paddle.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return {"IDMLProxyAnchorLoss": loss}
