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


class LinearTransformTeacher(nn.Layer):
    def __init__(self, qk_dim, t_shapes, keys):
        super().__init__()
        self.teacher_keys = keys
        self.t_shapes = [[1] + t_i for t_i in t_shapes]
        self.query_layer = nn.LayerList(
            [LinearBNReLU(t_shape[1], qk_dim) for t_shape in self.t_shapes])

    def forward(self, t_features_dict):
        g_t = [t_features_dict[key] for key in self.teacher_keys]
        bs = g_t[0].shape[0]
        channel_mean = [f_t.mean(3).mean(2) for f_t in g_t]
        spatial_mean = []
        for i in range(len(g_t)):
            c, h, w = g_t[i].shape[1:]
            spatial_mean.append(g_t[i].pow(2).mean(1).reshape([bs, h * w]))
        query = paddle.stack(
            [
                query_layer(
                    f_t, relu=False)
                for f_t, query_layer in zip(channel_mean, self.query_layer)
            ],
            axis=1)
        value = [F.normalize(f_s, axis=1) for f_s in spatial_mean]
        return {"query": query, "value": value}


class LinearTransformStudent(nn.Layer):
    def __init__(self, qk_dim, t_shapes, s_shapes, keys):
        super().__init__()
        self.student_keys = keys
        self.t_shapes = [[1] + t_i for t_i in t_shapes]
        self.s_shapes = [[1] + s_i for s_i in s_shapes]
        self.t = len(self.t_shapes)
        self.s = len(self.s_shapes)
        self.qk_dim = qk_dim
        self.n_t, self.unique_t_shapes = unique_shape(self.t_shapes)
        self.relu = nn.ReLU()
        self.samplers = nn.LayerList(
            [Sample(t_shape) for t_shape in self.unique_t_shapes])
        self.key_layer = nn.LayerList([
            LinearBNReLU(s_shape[1], self.qk_dim) for s_shape in self.s_shapes
        ])
        self.bilinear = LinearBNReLU(qk_dim, qk_dim * len(self.t_shapes))

    def forward(self, s_features_dict):
        g_s = [s_features_dict[key] for key in self.student_keys]
        bs = g_s[0].shape[0]
        channel_mean = [f_s.mean(3).mean(2) for f_s in g_s]
        spatial_mean = [sampler(g_s, bs) for sampler in self.samplers]

        key = paddle.stack(
            [
                key_layer(f_s)
                for key_layer, f_s in zip(self.key_layer, channel_mean)
            ],
            axis=1).reshape([-1, self.qk_dim])  # Bs x h
        bilinear_key = self.bilinear(
            key, relu=False).reshape([bs, self.s, self.t, self.qk_dim])
        value = [F.normalize(s_m, axis=2) for s_m in spatial_mean]
        return {"bilinear_key": bilinear_key, "value": value}


class Sample(nn.Layer):
    def __init__(self, t_shape):
        super().__init__()
        self.t_N, self.t_C, self.t_H, self.t_W = t_shape
        self.sample = nn.AdaptiveAvgPool2D((self.t_H, self.t_W))

    def forward(self, g_s, bs):
        g_s = paddle.stack(
            [
                self.sample(f_s.pow(2).mean(
                    1, keepdim=True)).reshape([bs, self.t_H * self.t_W])
                for f_s in g_s
            ],
            axis=1)
        return g_s
