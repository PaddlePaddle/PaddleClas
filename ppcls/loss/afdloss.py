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

import paddle.nn as nn
import paddle.nn.functional as F
import paddle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings
warnings.filterwarnings('ignore')


class LinearBNReLU(nn.Layer):
    def __init__(self, nin, nout):
        super().__init__()
        self.linear = nn.Linear(nin, nout)
        self.bn = nn.BatchNorm1D(nout)
        self.relu = nn.ReLU()

    def forward(self, x, relu=True):
        if relu:
            return self.relu(self.bn(self.linear(x)))
        return self.bn(self.linear(x))


def unique_shape(s_shapes):
    n_s = []
    unique_shapes = []
    n = -1
    for s_shape in s_shapes:
        if s_shape not in unique_shapes:
            unique_shapes.append(s_shape)
            n += 1
        n_s.append(n)
    return n_s, unique_shapes


class AFDLoss(nn.Layer):
    """
    AFDLoss
    https://www.aaai.org/AAAI21Papers/AAAI-9785.JiM.pdf
    https://github.com/clovaai/attention-feature-distillation
    """

    def __init__(self,
                 model_name_pair=["Student", "Teacher"],
                 student_keys=["bilinear_key", "value"],
                 teacher_keys=["query", "value"],
                 s_shapes=[[64, 16, 160], [128, 8, 160], [256, 4, 160],
                           [512, 2, 160]],
                 t_shapes=[[640, 48], [320, 96], [160, 192]],
                 qk_dim=128,
                 name="loss_afd"):
        super().__init__()
        assert isinstance(model_name_pair, list)
        self.model_name_pair = model_name_pair
        self.student_keys = student_keys
        self.teacher_keys = teacher_keys
        self.s_shapes = [[1] + s_i for s_i in s_shapes]
        self.t_shapes = [[1] + t_i for t_i in t_shapes]
        self.qk_dim = qk_dim
        self.n_t, self.unique_t_shapes = unique_shape(self.t_shapes)
        self.attention = Attention(self.qk_dim, self.t_shapes, self.s_shapes,
                                   self.n_t, self.unique_t_shapes)
        self.name = name

    def forward(self, predicts, batch):
        s_features_dict = predicts[self.model_name_pair[0]]
        t_features_dict = predicts[self.model_name_pair[1]]

        g_s = [s_features_dict[key] for key in self.student_keys]
        g_t = [t_features_dict[key] for key in self.teacher_keys]

        loss = self.attention(g_s, g_t)
        sum_loss = sum(loss)

        loss_dict = dict()
        loss_dict[self.name] = sum_loss

        return loss_dict


class Attention(nn.Layer):
    def __init__(self, qk_dim, t_shapes, s_shapes, n_t, unique_t_shapes):
        super().__init__()
        self.qk_dim = qk_dim
        self.n_t = n_t

        self.p_t = self.create_parameter(
            shape=[len(t_shapes), qk_dim],
            default_initializer=nn.initializer.XavierNormal())
        self.p_s = self.create_parameter(
            shape=[len(s_shapes), qk_dim],
            default_initializer=nn.initializer.XavierNormal())

    def forward(self, g_s, g_t):
        bilinear_key, h_hat_s_all = g_s
        query, h_t_all = g_t

        p_logit = paddle.matmul(self.p_t, self.p_s.t())

        logit = paddle.add(
            paddle.einsum('bstq,btq->bts', bilinear_key, query),
            p_logit) / np.sqrt(self.qk_dim)
        atts = F.softmax(logit, axis=2)  # b x t x s

        loss = []

        for i, (n, h_t) in enumerate(zip(self.n_t, h_t_all)):
            h_hat_s = h_hat_s_all[n]
            diff = self.cal_diff(h_hat_s, h_t, atts[:, i])
            loss.append(diff)
        return loss

    def cal_diff(self, v_s, v_t, att):
        diff = (v_s - v_t.unsqueeze(1)).pow(2).mean(2)
        diff = paddle.multiply(diff, att).sum(1).mean()
        return diff
