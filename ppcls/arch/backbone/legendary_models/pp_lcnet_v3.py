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

from __future__ import absolute_import, division, print_function

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle.nn.initializer import Constant, KaimingNormal
from paddle.nn import AdaptiveAvgPool2D, BatchNorm2D, Conv2D, Dropout, Hardsigmoid, Hardswish, Identity, Linear, ReLU
from paddle.regularizer import L2Decay

from ..base.theseus_layer import TheseusLayer
from ....utils.save_load import load_dygraph_pretrain, load_dygraph_pretrain_from_url

MODEL_URLS = {"PPLCNetV3": ""}

MODEL_STAGES_PATTERN = {
    "PPLCNetV3": ["blocks2", "blocks3", "blocks4", "blocks5", "blocks6"]
}

__all__ = list(MODEL_URLS.keys())

# Each element(list) represents a depthwise block, which is composed of k, in_c, out_c, s, use_se.
# k: kernel_size
# in_c: input channel number in depthwise block
# out_c: output channel number in depthwise block
# s: stride in depthwise block
# use_se: whether to use SE block

NET_CONFIG = {
    "blocks2":
    #k, in_c, out_c, s, use_se
    [[3, 16, 32, 1, False]],
    "blocks3": [[3, 32, 64, 2, False], [3, 64, 64, 1, False]],
    "blocks4": [[3, 64, 128, 2, False], [3, 128, 128, 1, False]],
    "blocks5": [[3, 128, 256, 2, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False], [5, 256, 256, 1, False],
                [5, 256, 256, 1, False]],
    "blocks6": [[5, 256, 512, 2, True], [5, 512, 512, 1, True],
                [5, 512, 512, 1, False], [5, 512, 512, 1, False]]
}


def make_divisible(v, divisor=16, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class LearnableAffineBlock(nn.Layer):
    def __init__(self,
                 scale_value=1.0,
                 bias_value=0.0,
                 lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.scale = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=scale_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("scale", self.scale)
        self.bias = self.create_parameter(
            shape=[1, ],
            default_initializer=Constant(value=bias_value),
            attr=ParamAttr(learning_rate=lr_mult * lab_lr))
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        return self.scale * x + self.bias


class ConvBNLayer(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 lr_mult=1.0):
        super().__init__()
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            groups=groups,
            weight_attr=ParamAttr(
                initializer=KaimingNormal(), learning_rate=lr_mult),
            bias_attr=False)

        self.bn = BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult),
            bias_attr=ParamAttr(
                regularizer=L2Decay(0.0), learning_rate=lr_mult))

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Act(nn.Layer):
    def __init__(self, act="hswish", lr_mult=1.0, lab_lr=0.1):
        super().__init__()
        if act == "hswish":
            self.act = Hardswish()
        else:
            assert act == "relu"
            self.act = ReLU()
        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        return self.lab(self.act(x))


class LearnableRepLayer(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 groups=1,
                 num_conv_branches=1,
                 lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.is_repped = False
        self.groups = groups
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.padding = (kernel_size - 1) // 2

        self.identity = BatchNorm2D(
            num_features=in_channels,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult)
        ) if out_channels == in_channels and stride == 1 else None

        self.conv_kxk = nn.LayerList([
            ConvBNLayer(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                groups=groups,
                lr_mult=lr_mult) for _ in range(self.num_conv_branches)
        ])

        self.conv_1x1 = ConvBNLayer(
            in_channels,
            out_channels,
            1,
            stride,
            groups=groups,
            lr_mult=lr_mult) if kernel_size > 1 else None

        self.lab = LearnableAffineBlock(lr_mult=lr_mult, lab_lr=lab_lr)
        self.act = Act(lr_mult=lr_mult, lab_lr=lab_lr)

    def forward(self, x):
        # for export
        if self.is_repped:
            out = self.lab(self.reparam_conv(x))
            if self.stride != 2:
                out = self.act(out)
            return out

        out = 0
        if self.identity is not None:
            out += self.identity(x)

        if self.conv_1x1 is not None:
            out += self.conv_1x1(x)

        for conv in self.conv_kxk:
            out += conv(x)

        out = self.lab(out)
        if self.stride != 2:
            out = self.act(out)
        return out

    def rep(self):
        if self.is_repped:
            return
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = Conv2D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups)
        self.reparam_conv.weight.set_value(kernel)
        self.reparam_conv.bias.set_value(bias)
        self.is_repped = True

    def _pad_kernel_1x1_to_kxk(self, kernel1x1, pad):
        if not isinstance(kernel1x1, paddle.Tensor):
            return 0
        else:
            return nn.functional.pad(kernel1x1, [pad, pad, pad, pad])

    def _get_kernel_bias(self):
        kernel_conv_1x1, bias_conv_1x1 = self._fuse_bn_tensor(self.conv_1x1)
        kernel_conv_1x1 = self._pad_kernel_1x1_to_kxk(kernel_conv_1x1,
                                                      self.kernel_size // 2)

        kernel_identity, bias_identity = self._fuse_bn_tensor(self.identity)

        kernel_conv_kxk = 0
        bias_conv_kxk = 0
        for conv in self.conv_kxk:
            kernel, bias = self._fuse_bn_tensor(conv)
            kernel_conv_kxk += kernel
            bias_conv_kxk += bias

        kernel_reparam = kernel_conv_kxk + kernel_conv_1x1 + kernel_identity
        bias_reparam = bias_conv_kxk + bias_conv_1x1 + bias_identity
        return kernel_reparam, bias_reparam

    def _fuse_bn_tensor(self, branch):
        if not branch:
            return 0, 0
        elif isinstance(branch, ConvBNLayer):
            kernel = branch.conv.weight
            running_mean = branch.bn._mean
            running_var = branch.bn._variance
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn._epsilon
        else:
            assert isinstance(branch, BatchNorm2D)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros(
                    (self.in_channels, input_dim, self.kernel_size,
                     self.kernel_size),
                    dtype=branch.weight.dtype)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, self.kernel_size // 2,
                                 self.kernel_size // 2] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std


class SELayer(TheseusLayer):
    def __init__(self, channel, reduction=4, lr_mult=1.0):
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2D(1)
        self.conv1 = Conv2D(
            in_channels=channel,
            out_channels=channel // reduction,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.relu = ReLU()
        self.conv2 = Conv2D(
            in_channels=channel // reduction,
            out_channels=channel,
            kernel_size=1,
            stride=1,
            padding=0,
            weight_attr=ParamAttr(learning_rate=lr_mult),
            bias_attr=ParamAttr(learning_rate=lr_mult))
        self.hardsigmoid = Hardsigmoid()

    def forward(self, x):
        identity = x
        x = self.avg_pool(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.hardsigmoid(x)
        x = paddle.multiply(x=identity, y=x)
        return x


class LCNetV3Block(TheseusLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 dw_size,
                 use_se=False,
                 conv_kxk_num=4,
                 lr_mult=1.0,
                 lab_lr=0.1):
        super().__init__()
        self.use_se = use_se
        self.dw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=dw_size,
            stride=stride,
            groups=in_channels,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)
        if use_se:
            self.se = SELayer(in_channels, lr_mult=lr_mult)
        self.pw_conv = LearnableRepLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            num_conv_branches=conv_kxk_num,
            lr_mult=lr_mult,
            lab_lr=lab_lr)

    def forward(self, x):
        x = self.dw_conv(x)
        if self.use_se:
            x = self.se(x)
        x = self.pw_conv(x)
        return x


class PPLCNetV3(TheseusLayer):
    def __init__(self,
                 stages_pattern,
                 scale=1.0,
                 conv_kxk_num=4,
                 class_num=1000,
                 dropout_prob=0.2,
                 class_expand=1280,
                 lr_mult_list=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 stride_list=[2, 1, 2, 2, 2, 2],
                 use_last_conv=True,
                 lab_lr=0.1,
                 return_patterns=None,
                 return_stages=None,
                 **kwargs):
        """PPLCNetV3

        Args:
            scale (float, optional): Network width scale. Defaults to 1.0.
            conv_kxk_num (int, optional): Number of re-parameter branches. Defaults to 4.
            class_num (int, optional): Number of classification. Defaults to 1000.
            dropout_prob (float, optional): Probability of dropout. Defaults to 0.2.
            class_expand (int, optional): Out channels of last conv. Defaults to 1280.
            lr_mult_list (list, optional): Learning rates for six different blocks. Defaults to [1.0, 1.0, 1.0, 1.0, 1.0, 1.0].
            stride_list (list, optional): Stride for six different blocks. Defaults to [2, 1, 2, 2, 2, 2].
            use_last_conv (bool, optional): Whether to use last conv block. Defaults to True.
            lab_lr (float, optional): Learning rate for all LearnableAffineLayer in model. Defaults to 0.1.
        """
        super().__init__()
        self.scale = scale
        self.class_expand = class_expand
        self.lr_mult_list = lr_mult_list
        self.use_last_conv = use_last_conv
        self.stride_list = stride_list
        self.net_config = NET_CONFIG

        assert isinstance(self.lr_mult_list, (
            list, tuple
        )), "lr_mult_list should be in (list, tuple) but got {}".format(
            type(self.lr_mult_list))
        assert len(self.lr_mult_list
                   ) == 6, "lr_mult_list length should be 6 but got {}".format(
                       len(self.lr_mult_list))

        assert isinstance(self.stride_list, (
            list, tuple
        )), "stride_list should be in (list, tuple) but got {}".format(
            type(self.stride_list))
        assert len(self.stride_list
                   ) == 6, "stride_list length should be 6 but got {}".format(
                       len(self.stride_list))

        for i, stride in enumerate(stride_list[1:]):
            self.net_config["blocks{}".format(i + 2)][0][3] = stride

        self.conv1 = ConvBNLayer(
            in_channels=3,
            out_channels=make_divisible(16 * scale),
            kernel_size=3,
            stride=stride_list[0],
            lr_mult=self.lr_mult_list[0])
        self.act1 = Act(lr_mult=self.lr_mult_list[0],
                        lab_lr=lab_lr) if stride_list[0] == 1 else Identity()

        self.blocks2 = nn.Sequential(* [
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[1],
                lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks2"])
        ])

        self.blocks3 = nn.Sequential(* [
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[2],
                lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks3"])
        ])

        self.blocks4 = nn.Sequential(* [
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[3],
                lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks4"])
        ])

        self.blocks5 = nn.Sequential(* [
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[4],
                lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks5"])
        ])

        self.blocks6 = nn.Sequential(* [
            LCNetV3Block(
                in_channels=make_divisible(in_c * scale),
                out_channels=make_divisible(out_c * scale),
                dw_size=k,
                stride=s,
                use_se=se,
                conv_kxk_num=conv_kxk_num,
                lr_mult=self.lr_mult_list[5],
                lab_lr=lab_lr) for i, (k, in_c, out_c, s, se) in enumerate(
                    self.net_config["blocks6"])
        ])

        self.avg_pool = AdaptiveAvgPool2D(1)
        if self.use_last_conv:
            self.last_conv = Conv2D(
                in_channels=make_divisible(self.net_config["blocks6"][-1][2] *
                                           scale),
                out_channels=self.class_expand,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False)
            self.act = Act(lab_lr=lab_lr)
            self.dropout = Dropout(p=dropout_prob, mode="downscale_in_infer")
        else:
            self.last_conv = None
        self.flatten = nn.Flatten(start_axis=1, stop_axis=-1)
        self.fc = Linear(
            self.class_expand if self.use_last_conv else
            make_divisible(self.net_config["blocks6"][-1][2] * scale),
            class_num)

        super().init_res(
            stages_pattern,
            return_patterns=return_patterns,
            return_stages=return_stages)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)

        x = self.blocks2(x)
        x = self.blocks3(x)
        x = self.blocks4(x)
        x = self.blocks5(x)
        x = self.blocks6(x)

        x = self.avg_pool(x)
        if self.last_conv is not None:
            x = self.last_conv(x)
            x = self.act(x)
            x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def _load_pretrained(pretrained, model, model_url, use_ssld):
    if pretrained is False:
        pass
    elif pretrained is True:
        load_dygraph_pretrain_from_url(model, model_url, use_ssld=use_ssld)
    elif isinstance(pretrained, str):
        load_dygraph_pretrain(model, pretrained)
    else:
        raise RuntimeError(
            "pretrained type is not available. Please use `string` or `boolean` type."
        )


def PPLCNetV3_base(pretrained=False, use_ssld=False, **kwargs):
    """PPLCNetV3_base

    Args:
        pretrained (bool, optional): If `True` load pretrained parameters, `False` otherwise. If 'str', means the path of the pretrained model. Defaults to False.
        use_ssld (bool, optional): Whether using distillation pretrained model when pretrained=True. Defaults to False.

    Returns:
        model: nn.Layer. Specific `PPLCNetV3_base` model depends on args.        
    """
    model = PPLCNetV3(
        scale=0.95,
        conv_kxk_num=4,
        stages_pattern=MODEL_STAGES_PATTERN["PPLCNetV3"],
        **kwargs)
    _load_pretrained(pretrained, model, MODEL_URLS["PPLCNetV3"], use_ssld)
    return model


if __name__ == "__main__":
    import paddle
    from ....utils import logger
    from ....utils.logger import init_logger

    init_logger()

    paddle.seed(123)
    input_x = paddle.randn((1, 3, 224, 224), dtype="float32")
    net = PPLCNetV3_base()
    state_dict = paddle.load("./PPLCNetV3/refactor.pdparams")

    # pretrained = paddle.load("./PPLCNetV3/best_model.pdparams")
    # pretrained.pop('conv1.hardswish.scale')
    # pretrained.pop('conv1.hardswish.bias')

    # pretrained = paddle.load("./PPLCNetV3/refactor.pdparams")
    # pretrained_keys = list(pretrained.keys())
    # pretrained_values = list(pretrained.values())

    # state_dict = net.state_dict()
    # assert len(state_dict) == len(pretrained)
    # for i, k in enumerate(state_dict):
    #     state_dict[k] = pretrained_values[i]
    # print(k, pretrained_keys[i])

    net.set_dict(state_dict)
    # paddle.save(net.state_dict(), './PPLCNetV3/refactor.pdparams')

    for layer in net.sublayers():
        if hasattr(layer, "rep") and not getattr(layer, "is_repped"):
            layer.rep()

    net.eval()
    output = net(input_x)
    print(output[0][:10])
