#copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

import paddle
import paddle.nn as nn


class DSHSDLoss(nn.Layer):
    """
    # DSHSD(IEEE ACCESS 2019)
    # paper [Deep Supervised Hashing Based on Stable Distribution](https://ieeexplore.ieee.org/document/8648432/)
    # code reference: https://github.com/swuxyj/DeepHash-pytorch/blob/master/DSHSD.py
    """

    def __init__(self, alpha, multi_label=False):
        super(DSHSDLoss, self).__init__()
        self.alpha = alpha
        self.multi_label = multi_label

    def forward(self, input, label):
        features = input["features"]
        logits = input["logits"]

        features_temp1 = paddle.unsqueeze(features, 1)
        features_temp2 = paddle.unsqueeze(features, 0)
        dist = features_temp1 - features_temp2
        dist = paddle.square(dist)
        dist = paddle.sum(dist, axis=2)

        n_class = logits.shape[1]
        labels = paddle.nn.functional.one_hot(label, n_class)
        labels = labels.squeeze().astype("float32")

        s = paddle.matmul(labels, labels, transpose_y=True)
        s = (s == 0).astype("float32")
        margin = 2 * features.shape[1]
        Ld = (1 - s) / 2 * dist + s / 2 * (margin - dist).clip(min=0)
        Ld = Ld.mean()

        if self.multi_label:
            Lc_temp = (1 + (-logits).exp()).log()
            Lc = (logits - labels * logits + Lc_temp).sum(axis=1)
        else:
            probs = paddle.nn.functional.softmax(logits)
            Lc = (-probs.log() * labels).sum(axis=1)
        Lc = Lc.mean()

        loss = Lc + Ld * self.alpha
        return {"dshsdloss": loss}


class LCDSHLoss(nn.Layer):
    """
    # paper [Locality-Constrained Deep Supervised Hashing for Image Retrieval](https://www.ijcai.org/Proceedings/2017/0499.pdf)
    # code reference: https://github.com/swuxyj/DeepHash-pytorch/blob/master/LCDSH.py
    """

    def __init__(self, n_class, _lambda):
        super(LCDSHLoss, self).__init__()
        self._lambda = _lambda
        self.n_class = n_class

    def forward(self, input, label):
        features = input["features"]
        labels = paddle.nn.functional.one_hot(label, self.n_class)
        labels = labels.squeeze().astype("float32")

        s = paddle.matmul(labels, labels, transpose_y=True)
        s = 2 * (s > 0).astype("float32") - 1

        inner_product = paddle.matmul(features, features, transpose_y=True)
        inner_product = inner_product * 0.5
        inner_product = inner_product.clip(min=-50, max=50)
        L1 = paddle.log(1 + paddle.exp(-s * inner_product))
        L1 = L1.mean()

        binary_features = features.sign()

        inner_product_ = paddle.matmul(
            binary_features, binary_features, transpose_y=True)
        inner_product_ = inner_product_ * 0.5
        sigmoid = paddle.nn.Sigmoid()
        L2 = (sigmoid(inner_product) - sigmoid(inner_product_)).pow(2)
        L2 = L2.mean()

        loss = L1 + self._lambda * L2
        return {"lcdshloss": loss}


class DCHLoss(paddle.nn.Layer):
    """
    # paper [Deep Cauchy Hashing for Hamming Space Retrieval]
    URL:(http://ise.thss.tsinghua.edu.cn/~mlong/doc/deep-cauchy-hashing-cvpr18.pdf)
    # code reference: https://github.com/swuxyj/DeepHash-pytorch/blob/master/DCH.py
    """

    def __init__(self, gamma, _lambda, n_class):
        super(DCHLoss, self).__init__()
        self.gamma = gamma
        self._lambda = _lambda
        self.n_class = n_class

    def distance(self, feature_i, feature_j):
        assert feature_i.shape[1] == feature_j.shape[
            1], "feature len of feature_i and feature_j is different, please check whether the featurs are right"
        K = feature_i.shape[1]
        inner_product = paddle.matmul(feature_i, feature_j, transpose_y=True)

        len_i = feature_i.pow(2).sum(axis=1, keepdim=True).pow(0.5)
        len_j = feature_j.pow(2).sum(axis=1, keepdim=True).pow(0.5)
        norm = paddle.matmul(len_i, len_j, transpose_y=True)
        cos = inner_product / norm.clip(min=0.0001)
        dist = (1 - cos.clip(max=0.99)) * K / 2
        return dist

    def forward(self, input, label):
        features = input["features"]
        labels = paddle.nn.functional.one_hot(label, self.n_class)
        labels = labels.squeeze().astype("float32")

        s = paddle.matmul(labels, labels, transpose_y=True).astype("float32")
        if (1 - s).sum() != 0 and s.sum() != 0:
            positive_w = s * s.numel() / s.sum()
            negative_w = (1 - s) * s.numel() / (1 - s).sum()
            w = positive_w + negative_w
        else:
            w = 1

        dist_matric = self.distance(features, features)
        cauchy_loss = w * (s * paddle.log(dist_matric / self.gamma) +
                           paddle.log(1 + self.gamma / dist_matric))

        all_one = paddle.ones_like(features, dtype="float32")
        dist_to_one = self.distance(features.abs(), all_one)
        quantization_loss = paddle.log(1 + dist_to_one / self.gamma)

        loss = cauchy_loss.mean() + self._lambda * quantization_loss.mean()
        return {"dchloss": loss}
