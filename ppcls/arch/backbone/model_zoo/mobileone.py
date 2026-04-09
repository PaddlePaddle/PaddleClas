# copyright (c) 2026 PaddlePaddle Authors. All Rights Reserve.
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
#
# reference: https://arxiv.org/abs/2206.04040

from typing import List, Optional, Tuple
import copy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ....utils.save_load import load_dygraph_pretrain

MODEL_URLS = {
    "MobileOne_S0": "",
    "MobileOne_S1": "",
    "MobileOne_S2": "",
    "MobileOne_S3": "",
    "MobileOne_S4": "",
}

__all__ = list(MODEL_URLS.keys())


class SEBlock(nn.Layer):
    """Squeeze-and-Excitation block."""

    def __init__(self, in_channels, rd_ratio=0.0625):
        super().__init__()
        self.reduce = nn.Conv2D(
            in_channels=in_channels,
            out_channels=int(in_channels * rd_ratio),
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )
        self.expand = nn.Conv2D(
            in_channels=int(in_channels * rd_ratio),
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias_attr=True,
        )

    def forward(self, x):
        x_se = F.adaptive_avg_pool2d(x, output_size=1)
        x_se = self.reduce(x_se)
        x_se = F.relu(x_se)
        x_se = self.expand(x_se)
        x_se = F.sigmoid(x_se)
        return x * x_se


class MobileOneBlock(nn.Layer):
    """Train-time multi-branch block with deploy-time re-parameterization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
    ):
        super().__init__()
        self.inference_mode = inference_mode
        self.is_repped = inference_mode

        self.groups = groups
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches

        self.se = SEBlock(out_channels) if use_se else nn.Identity()
        self.activation = nn.ReLU()

        if inference_mode:
            self.reparam_conv = nn.Conv2D(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias_attr=True,
            )
            return

        self.rbr_skip = (
            nn.BatchNorm2D(num_features=in_channels)
            if out_channels == in_channels and stride == 1
            else None
        )

        rbr_conv = []
        for _ in range(self.num_conv_branches):
            rbr_conv.append(self._conv_bn(kernel_size=kernel_size, padding=padding))
        self.rbr_conv = nn.LayerList(rbr_conv)

        self.rbr_scale = None
        if kernel_size > 1:
            self.rbr_scale = self._conv_bn(kernel_size=1, padding=0)

    def forward(self, x):
        if self.is_repped:
            return self.activation(self.se(self.reparam_conv(x)))

        identity_out = 0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        scale_out = 0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        out = scale_out + identity_out
        for branch in self.rbr_conv:
            out += branch(x)

        return self.activation(self.se(out))

    def reparameterize(self):
        self.re_parameterize()

    def re_parameterize(self):
        if self.is_repped:
            return

        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias_attr=True,
        )
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)

        del self.rbr_conv
        if self.rbr_scale is not None:
            del self.rbr_scale
        if self.rbr_skip is not None:
            del self.rbr_skip

        self.inference_mode = True
        self.is_repped = True

    def _get_kernel_bias(self):
        kernel_scale, bias_scale = 0, 0
        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            pad = self.kernel_size // 2
            kernel_scale = F.pad(
                kernel_scale, [pad, pad, pad, pad], mode="constant", value=0.0
            )

        kernel_identity, bias_identity = 0, 0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        kernel_conv, bias_conv = 0, 0
        for branch in self.rbr_conv:
            _kernel, _bias = self._fuse_bn_tensor(branch)
            kernel_conv += _kernel
            bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch):
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros(
                    shape=[
                        self.in_channels,
                        input_dim,
                        self.kernel_size,
                        self.kernel_size,
                    ],
                    dtype=branch.weight.dtype,
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2, self.kernel_size // 2] = 1.0
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon

        std = paddle.sqrt(running_var + eps)
        t = paddle.reshape(gamma / std, [-1, 1, 1, 1])
        return kernel * t, beta - running_mean * gamma / std

    def _conv_bn(self, kernel_size, padding):
        layer = nn.Sequential()
        layer.add_sublayer(
            "conv",
            nn.Conv2D(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                groups=self.groups,
                bias_attr=False,
            ),
        )
        layer.add_sublayer("bn", nn.BatchNorm2D(num_features=self.out_channels))
        return layer


class MobileOne(nn.Layer):
    """MobileOne backbone for image classification."""

    def __init__(
        self,
        num_blocks_per_stage=None,
        class_num=1000,
        num_classes=None,
        width_multipliers=None,
        inference_mode=False,
        use_se=False,
        num_conv_branches=1,
    ):
        super().__init__()
        if num_blocks_per_stage is None:
            num_blocks_per_stage = [2, 8, 10, 1]
        if width_multipliers is None:
            raise ValueError("`width_multipliers` must be provided and contain 4 elements.")
        if len(width_multipliers) != 4:
            raise ValueError("`width_multipliers` should contain 4 elements.")
        if num_classes is not None:
            class_num = num_classes

        self.class_num = class_num
        self.inference_mode = inference_mode
        self.in_planes = min(64, int(64 * width_multipliers[0]))
        self.use_se = use_se
        self.num_conv_branches = num_conv_branches

        self.stage0 = MobileOneBlock(
            in_channels=3,
            out_channels=self.in_planes,
            kernel_size=3,
            stride=2,
            padding=1,
            inference_mode=self.inference_mode,
        )
        self.stage1 = self._make_stage(
            int(64 * width_multipliers[0]), num_blocks_per_stage[0], num_se_blocks=0
        )
        self.stage2 = self._make_stage(
            int(128 * width_multipliers[1]), num_blocks_per_stage[1], num_se_blocks=0
        )
        self.stage3 = self._make_stage(
            int(256 * width_multipliers[2]),
            num_blocks_per_stage[2],
            num_se_blocks=int(num_blocks_per_stage[2] // 2) if use_se else 0,
        )
        self.stage4 = self._make_stage(
            int(512 * width_multipliers[3]),
            num_blocks_per_stage[3],
            num_se_blocks=num_blocks_per_stage[3] if use_se else 0,
        )
        self.gap = nn.AdaptiveAvgPool2D(output_size=1)
        self.linear = nn.Linear(int(512 * width_multipliers[3]), class_num)

    def _make_stage(self, planes, num_blocks, num_se_blocks):
        strides = [2] + [1] * (num_blocks - 1)
        blocks = []
        for idx, stride in enumerate(strides):
            use_se = False
            if num_se_blocks > num_blocks:
                raise ValueError("Number of SE blocks cannot exceed number of layers.")
            if idx >= (num_blocks - num_se_blocks):
                use_se = True

            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=self.in_planes,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.in_planes,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            blocks.append(
                MobileOneBlock(
                    in_channels=self.in_planes,
                    out_channels=planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                    inference_mode=self.inference_mode,
                    use_se=use_se,
                    num_conv_branches=self.num_conv_branches,
                )
            )
            self.in_planes = planes
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stage0(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = paddle.flatten(x, start_axis=1)
        x = self.linear(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld=False):
    if pretrained is False:
        pass
    elif pretrained is True:
        if not model_url:
            raise RuntimeError(
                "No official pretrained URL configured for this MobileOne variant. "
                "Please pass a local `.pdparams` path to `pretrained`."
            )
        load_dygraph_pretrain(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def MobileOne_S0(pretrained=False, use_ssld=False, **kwargs):
    model = MobileOne(
        width_multipliers=(0.75, 1.0, 1.0, 2.0),
        num_conv_branches=4,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["MobileOne_S0"], use_ssld)
    return model


def MobileOne_S1(pretrained=False, use_ssld=False, **kwargs):
    model = MobileOne(width_multipliers=(1.5, 1.5, 2.0, 2.5), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileOne_S1"], use_ssld)
    return model


def MobileOne_S2(pretrained=False, use_ssld=False, **kwargs):
    model = MobileOne(width_multipliers=(1.5, 2.0, 2.5, 4.0), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileOne_S2"], use_ssld)
    return model


def MobileOne_S3(pretrained=False, use_ssld=False, **kwargs):
    model = MobileOne(width_multipliers=(2.0, 2.5, 3.0, 4.0), **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["MobileOne_S3"], use_ssld)
    return model


def MobileOne_S4(pretrained=False, use_ssld=False, **kwargs):
    model = MobileOne(
        width_multipliers=(3.0, 3.5, 3.5, 4.0),
        use_se=True,
        **kwargs
    )
    _load_pretrained(pretrained, model, MODEL_URLS["MobileOne_S4"], use_ssld)
    return model


def mobileone(class_num=1000, inference_mode=False, variant="s0", **kwargs):
    variant = variant.lower()
    if variant == "s0":
        return MobileOne_S0(
            class_num=class_num, inference_mode=inference_mode, **kwargs
        )
    if variant == "s1":
        return MobileOne_S1(
            class_num=class_num, inference_mode=inference_mode, **kwargs
        )
    if variant == "s2":
        return MobileOne_S2(
            class_num=class_num, inference_mode=inference_mode, **kwargs
        )
    if variant == "s3":
        return MobileOne_S3(
            class_num=class_num, inference_mode=inference_mode, **kwargs
        )
    if variant == "s4":
        return MobileOne_S4(
            class_num=class_num, inference_mode=inference_mode, **kwargs
        )
    raise ValueError(f"Unsupported MobileOne variant: {variant}")


def reparameterize_model(model):
    model = copy.deepcopy(model)
    for module in model.sublayers():
        if hasattr(module, "re_parameterize"):
            module.re_parameterize()
        elif hasattr(module, "reparameterize"):
            module.reparameterize()
    return model
