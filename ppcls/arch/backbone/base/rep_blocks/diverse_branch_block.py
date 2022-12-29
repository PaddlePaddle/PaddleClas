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

import importlib
from functools import reduce

import paddle.nn as nn

from .....utils import logger
from ..theseus_layer import TheseusLayer
from .branches import ConvKxK, Conv1x1, Conv1x1_KxK, Conv1x1_AVG


class RepBlock(TheseusLayer):
    def __init__(self):
        super().__init__()

    def re_parameterize(self):
        raise NotImplementedError("")


class DiverseBranchBlock(RepBlock):
    def __init__(self,
                 config,
                 act=None,
                 data_format="NCHW",
                 is_repped=False,
                 single_init=False):
        super().__init__()
        in_channels, out_channels, kernel_size, stride, padding, dilation, groups, internal_channels, with_bn = config[
            "in_channels"], config["out_channels"], config[
                "kernel_size"], config["stride"], config["padding"], config[
                    "dilation"], config["groups"], config[
                        "internal_channels"], config["with_bn"]
        # assert padding == kernel_size // 2

        # TODO(gaotingquan): support channel last format(NHWC) for resnet, etc.
        if data_format != "NCHW":
            msg = "DiverseBranchBlock only support NCHW format now!"
            logger.error(msg)
            raise Exception(msg)

        self.config = config
        self.is_repped = is_repped

        branches_config = {
            "ConvKxK": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "groups": groups,
                "with_bn": with_bn
            },
            "Conv1x1": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "stride": stride,
                "groups": groups,
                "with_bn": with_bn
            },
            "Conv1x1_AVG": {
                "in_channels": in_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "groups": groups,
                "with_bn": with_bn
            },
            "Conv1x1_KxK": {
                "in_channels": in_channels,
                "internal_channels": internal_channels,
                "out_channels": out_channels,
                "kernel_size": kernel_size,
                "stride": stride,
                "padding": padding,
                "groups": groups,
                "with_bn": with_bn
            }
        }

        self.branch_dict = nn.LayerDict()
        for branch_name in branches_config:
            mod = importlib.import_module(__name__)
            block = getattr(mod, branch_name)(**branches_config[branch_name])
            self.branch_dict[branch_name] = block
        self.act = act

        #  The experiments reported in the paper used the default initialization of bn.weight (all as 1). But changing the initialization may be useful in some cases.
        if single_init:
            #  Initialize the bn.weight of dbb_origin as 1 and others as 0. This is not the default setting.
            self.single_init()

    def forward(self, x):
        if self.is_repped:
            out = self.repped_conv(x)
            if self.act:
                out = self.act(out)
            return out

        out = 0
        for branch_name in self.branch_dict:
            out += self.branch_dict[branch_name](x)
        if self.act:
            out = self.act(out)

        return out

    def single_init(self):
        for branch_name in self.branch_dict:
            if branch_name == "ConvKxK":
                nn.init.constant_(self.branch_dict[branch_name].bn.weight, 1.0)
            else:
                nn.init.constant_(self.branch_dict[branch_name].bn.weight, 0.0)

    def re_parameterize(self):
        if self.is_repped:
            return
        w, b = self.get_actual_kernel()

        self.repped_conv = nn.Conv2D(
            in_channels=self.config["in_channels"],
            out_channels=self.config["out_channels"],
            kernel_size=self.config["kernel_size"],
            stride=self.config["stride"],
            padding=self.config["padding"],
            dilation=self.config["dilation"],
            groups=self.config["groups"],
            bias_attr=True)

        self.repped_conv.weight.set_value(w)
        self.repped_conv.bias.set_value(b)
        # TODO(gaotingquan): is OK ?
        # self.branch_list = []
        del self.branch_list
        self.is_repped = True


# TODO(gaotingquan): support vd
def resnet2dbb(model):
    def handle_func(layer, pattern):
        config = {
            "in_channels": layer.conv._in_channels,
            "out_channels": layer.conv._out_channels,
            "kernel_size": layer.conv._kernel_size,
            "stride": layer.conv._stride,
            "padding": layer.conv._padding,
            "dilation": layer.conv._dilation,
            "groups": layer.conv._groups,
            "internal_channels": layer.conv._in_channels,
            "with_bn": True
        }
        dbb_layer = DiverseBranchBlock(
            config=config,
            act=nn.ReLU() if layer.act else None,
            data_format=layer.data_format)
        return dbb_layer

    if model.block_type == "BasicBlock":
        name_list = [[f"blocks[{i}].conv0", f"blocks[{i}].conv1"]
                     for i in range(len(model.blocks))]
    elif model.block_type == "BottleneckBlock":
        name_list = [[
            f"blocks[{i}].conv0", f"blocks[{i}].conv1", f"blocks[{i}].conv2"
        ] for i in range(len(model.blocks))]
    else:
        raise Exception()
    layer_name_list = reduce(lambda x, y: x + y, name_list)

    model.upgrade_sublayer(
        layer_name_pattern=layer_name_list, handle_func=handle_func)
