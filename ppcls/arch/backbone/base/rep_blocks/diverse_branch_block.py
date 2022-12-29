import importlib
import paddle
import paddle.nn as nn
# from ..rep_blocks import RepBlock
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
                paddle.nn.init.constant_(
                    self.branch_dict[branch_name].bn.weight, 1.0)
            else:
                paddle.nn.init.constant_(
                    self.branch_dict[branch_name].bn.weight, 0.0)

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
