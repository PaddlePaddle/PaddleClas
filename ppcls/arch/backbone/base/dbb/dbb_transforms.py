# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
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

# reference: https://arxiv.org/abs/2103.13425, https://github.com/DingXiaoH/DiverseBranchBlock

import numpy as np
import paddle
import paddle.nn.functional as F


def transI_fusebn(kernel, bn):
    gamma = bn.weight
    std = (bn._variance + bn._epsilon).sqrt()
    return kernel * (
        (gamma / std).reshape([-1, 1, 1, 1])), bn.bias - bn._mean * gamma / std


def transII_addbranch(kernels, biases):
    return sum(kernels), sum(biases)


def transIII_1x1_kxk(k1, b1, k2, b2, groups):
    if groups == 1:
        k = F.conv2d(k2, k1.transpose([1, 0, 2, 3]))
        b_hat = (k2 * b1.reshape([1, -1, 1, 1])).sum((1, 2, 3))
    else:
        k_slices = []
        b_slices = []
        k1_T = k1.transpose([1, 0, 2, 3])
        k1_group_width = k1.shape[0] // groups
        k2_group_width = k2.shape[0] // groups
        for g in range(groups):
            k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) *
                              k1_group_width, :, :]
            k2_slice = k2[g * k2_group_width:(g + 1) * k2_group_width, :, :, :]
            k_slices.append(F.conv2d(k2_slice, k1_T_slice))
            b_slices.append((k2_slice * b1[g * k1_group_width:(
                g + 1) * k1_group_width].reshape([1, -1, 1, 1])).sum((1, 2, 3
                                                                      )))
        k, b_hat = transIV_depthconcat(k_slices, b_slices)
    return k, b_hat + b2


def transIV_depthconcat(kernels, biases):
    return paddle.cat(kernels, axis=0), paddle.cat(biases)


def transV_avg(channels, kernel_size, groups):
    input_dim = channels // groups
    k = paddle.zeros((channels, input_dim, kernel_size, kernel_size))
    k[np.arange(channels), np.tile(np.arange(input_dim),
                                   groups), :, :] = 1.0 / kernel_size**2
    return k


# This has not been tested with non-square kernels (kernel.shape[2] != kernel.shape[3]) nor even-size kernels
def transVI_multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.shape[2]) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.shape[3]) // 2
    return F.pad(
        kernel,
        [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])
