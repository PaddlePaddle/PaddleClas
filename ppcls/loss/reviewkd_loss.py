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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils.initializer import kaiming_normal_, kaiming_uniform_


class ABF(nn.Layer):
    def __init__(self, in_channel, mid_channel, out_channel, fuse):
        super().__init__()
        self.conv1 = nn.Conv2D(
            in_channel, mid_channel, kernel_size=1, bias_attr=False)
        self.conv1_bn = nn.BatchNorm2D(mid_channel)

        self.conv2 = nn.Conv2D(
            mid_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv2_bn = nn.BatchNorm2D(out_channel)
        if fuse:
            self.att_conv = nn.Sequential(
                nn.Conv2D(
                    mid_channel * 2, 2, kernel_size=1),
                nn.Sigmoid(), )
        else:
            self.att_conv = None

        self.init_params()

    def init_params(self, ):
        kaiming_uniform_(self.conv1.weight, a=1)
        kaiming_uniform_(self.conv2.weight, a=1)
        if self.att_conv is not None:
            kaiming_normal_(
                self.att_conv[0].weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x, y=None):
        n, _, h, w = x.shape
        # transform student features
        x = self.conv1(x)
        x = self.conv1_bn(x)
        if self.att_conv is not None:
            # upsample residual features
            y = F.interpolate(y, (h, w), mode="nearest")
            # fusion
            z = paddle.concat([x, y], axis=1)
            z = self.att_conv(z)
            x = (x * z[:, 0].reshape([n, 1, h, w]) + y * z[:, 1].reshape(
                [n, 1, h, w]))
        y = self.conv2(x)
        y = self.conv2_bn(y)
        return y, x


class HCL(nn.Layer):
    def __init__(self, mode="avg"):
        super().__init__()
        assert mode in ["max", "avg"]
        self.mode = mode

    def forward(self, fstudent, fteacher):
        loss_all = 0.0
        for fs, ft in zip(fstudent, fteacher):
            h = fs.shape[2]
            loss = F.mse_loss(fs, ft)
            cnt = 1.0
            tot = 1.0
            for l in [4, 2, 1]:
                if l >= h:
                    continue
                if self.mode == "max":
                    tmpfs = F.adaptive_max_pool2d(fs, (l, l))
                    tmpft = F.adaptive_max_pool2d(ft, (l, l))
                else:
                    tmpfs = F.adaptive_avg_pool2d(fs, (l, l))
                    tmpft = F.adaptive_avg_pool2d(ft, (l, l))

                cnt /= 2.0
                loss += F.mse_loss(tmpfs, tmpft) * cnt
                tot += cnt
            loss = loss / tot
            loss_all = loss_all + loss
        return loss_all


class ReviewKDLoss(nn.Layer):
    """Paddle version of `Distilling Knowledge via Knowledge Review`
    ReviewKDLoss
    Reference: https://arxiv.org/pdf/2104.09044.pdf
    Code was heavily based on https://github.com/dvlab-research/ReviewKD
    """

    def __init__(self,
                 student_channels,
                 teacher_channels,
                 mid_channel=256,
                 hcl_mode="avg"):
        super().__init__()
        abfs = nn.LayerList()

        for idx, student_channel in enumerate(student_channels):
            abfs.append(
                ABF(student_channel, mid_channel, teacher_channels[idx], idx <
                    len(student_channels) - 1))
        self.abfs = abfs[::-1]

        self.hcl = HCL(mode=hcl_mode)

    def forward(self, student_features, teacher_features):
        x = student_features[::-1]
        results = []
        out_features, res_features = self.abfs[0](x[0])
        results.append(out_features)
        for features, abf in zip(x[1:], self.abfs[1:]):
            out_features, res_features = abf(features, res_features)
            results.insert(0, out_features)
        loss = self.hcl(results, teacher_features)
        return loss
