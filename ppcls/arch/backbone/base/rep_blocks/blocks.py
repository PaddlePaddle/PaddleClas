import paddle
from paddle import nn

from ..theseus_layer import TheseusLayer


class RepBlock(TheseusLayer):
    def __init__(self):
        super().__init__()

    def re_parameterize(self):
        raise NotImplementedError("")


class ConvBN(RepBlock):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False,
                 with_bn=True):
        super().__init__()
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)

        self.bn = nn.BatchNorm2D(
            num_features=out_channels) if with_bn else nn.Identity()

        self.deployed = False
        self.deploy_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias_attr)

    def forward(self, x):
        if self.deployed:
            return self.deploy_conv(x)
        x = self.conv(x)
        x = self.bn(x)
        return output

    def re_parameterize(self):
        if not isinstance(self.bn, nn.BatchNorm2D):
            kernel_hat = self.conv.kernel
            bias_hat = self.conv.bias
        else:
            gamma = self.bn.weight
            std = (self.bn._variance + self.bn._epsilon).sqrt()
            bias = -self.bn._mean
            if self.conv.bias is not None:
                bias += self.conv.bias

            kernel_hat = self.conv.kernel * (
                (gamma / std).reshape([-1, 1, 1, 1]))
            bias_hat = self.bn.bias + bias * gamma / std

        self.deploy_conv.weight.set_value(kernel_hat)
        if bias_hat is not None:
            self.deploy_conv.bias.set_value(bias_hat)
        self.deployed = True


class ConvKxK(RepBlock):
    def __init__(self, config):
        self.block = ConvBN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=config["kernel_size"],
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
            groups=config["groups"],
            bias_attr=config["bias_attr"],
            with_bn=config["with_bn"])

    def forward(self, x):
        return self.block(x)

    def re_parameterize(self):
        self.block.re_parameterize()


class Conv1xK(RepBlock):
    def __init__(self, config):
        self.block = ConvBN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=(1, config["kernel_size"]),
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
            groups=config["groups"],
            bias_attr=config["bias_attr"],
            with_bn=config["with_bn"])

    def forward(self, x):
        return self.block(x)

    def re_parameterize(self):
        self.block.re_parameterize()


class ConvKx1(RepBlock):
    def __init__(self, config):
        self.block = ConvBN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=(config["kernel_size"], 1),
            stride=config["stride"],
            padding=config["padding"],
            dilation=config["dilation"],
            groups=config["groups"],
            bias_attr=config["bias_attr"],
            with_bn=config["with_bn"])

    def forward(self, x):
        return self.block(x)

    def re_parameterize(self):
        self.block.re_parameterize()


class Conv1x1(RepBlock):
    def __init__(self, config):
        self.block = ConvBN(
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            kernel_size=1,
            stride=config["stride"],
            padding=0,
            groups=config["groups"],
            bias_attr=config["bias_attr"],
            with_bn=config["with_bn"])

    def forward(self, x):
        return self.block(x)

    def re_parameterize(self):
        self.block.re_parameterize()


class Conv1x1_KxK(RepBlock):
    def __init__(self, config):
        in_channels, internal_channels, out_channels, kernel_size, stride, padding, groups, with_bn = config[
            "in_channels"], config["internal_channels"], config[
                "out_channels"], config["kernel_size"], config[
                    "stride"], config["padding"], config["groups"], config[
                        "with_bn"]
        self.conv1x1 = Conv1x1(
            in_channels=in_channels,
            out_channels=internal_channels,
            groups=groups,
            with_bn=True)
        self.convkxk = ConvKxK(
            in_channels=internal_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            groups=groups,
            with_bn=True)

        self.deployed = False
        self.deploy_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2),
            groups=groups)

    def forward(self, x):
        if self.deployed:
            return self.deploy_conv(x)
        x = self.conv1x1(x)
        x = self.convkxk(x)
        return x

    def re_parameterize(self):
        self.conv1x1.re_parameterize()
        self.convkxk.re_parameterize()
        k1 = self.conv1x1.deploy_conv.weight
        b1 = self.conv1x1.deploy_conv.bias
        k2 = self.convkxk.deploy_conv.weight
        b2 = self.convkxk.deploy_conv.bias

        if self.groups == 1:
            kernel_hat = F.conv2d(k2, k1.transpose([1, 0, 2, 3]))
            bias_hat = (k2 * b1.reshape([1, -1, 1, 1])).sum((1, 2, 3)) + b2
        else:
            k_slices = []
            b_slices = []
            k1_T = k1.transpose([1, 0, 2, 3])
            k1_group_width = k1.shape[0] // groups
            k2_group_width = k2.shape[0] // groups
            for g in range(groups):
                k1_T_slice = k1_T[:, g * k1_group_width:(g + 1) *
                                  k1_group_width, :, :]
                k2_slice = k2[g * k2_group_width:(g + 1) *
                              k2_group_width, :, :, :]
                k_slices.append(F.conv2d(k2_slice, k1_T_slice))
                b_slices.append((k2_slice * b1[g * k1_group_width:(
                    g + 1) * k1_group_width].reshape([1, -1, 1, 1])).sum((1, 2,
                                                                          3)))
            kernel_hat = paddle.concat(k_slices)
            bias_hat = paddle.concat(b_slices) + b2

        self.deploy_conv.weight.set_value(kernel_hat)
        self.deploy_conv.bias.set_value(bias_hat)
        self.deployed = True


# TODO(gaotingquan): is it valid when stride == 2
class Conv1x1_AVG(RepBlock):
    def __init__(self, config):
        in_channels, out_channels, kernel_size, groups, bias_attr, with_bn = config[
            "in_channels"], config["out_channels"], config[
                "kernel_size"], config["groups"], config["bias_attr"], config[
                    "with_bn"]

        self.conv = ConvBN(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=groups,
            bias_attr=bias_attr,
            with_bn=with_bn)
        self.avg = nn.AvgPool2D(
            kernel_size=kernel_size, stride=stride, padding=0)
        self.bn = nn.BatchNorm2D(num_features=out_channels)

        self.deployed = False
        self.deploy_conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size // 2),
            groups=groups)

    def forward(self, x):
        if self.deployed:
            return self.deploy_conv(x)
        x = self.conv(x)
        x = self.avg(x)
        x = self.bn(x)
        return x

    def re_parameterize(self):
        pass


class IdentityBN(RepBlock):
    def __init__(self, config):
        if config["in_channels"] != config["out_channels"]:
            self.invalid = True
        else:
            self.invalid = False

        num_features, kernel_size, groups = config["out_channels"], config[
            "kernel_size"], config["groups"]
        self.bn = nn.BatchNorm2D(num_features=num_features)
        self.num_features = num_features
        self.kernel_size = kernel_size
        self.groups = groups

        self.deployed = False
        self.deploy_conv = nn.Conv2D(
            in_channels=num_features,
            out_channels=num_features,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2),
            groups=groups)

    def forward(self, x):
        if self.invalid:
            return 0
        if self.deployed:
            return self.deploy_conv(x)
        return self.bn(x)

    def re_parameterize(self):
        if self.invalid:
            return
        input_dim = self.num_features // self.groups
        # TODO(gaotingquan): paddle.zeros()
        kernel_value = np.zeros(
            (self.num_features, input_dim, self.kernel_size, self.kernel_size),
            dtype=np.float32)
        center_idx = self.kernel_size // 2
        for i in range(in_channels):
            kernel_value[i, i % input_dim, center_idx, center_idx] = 1
        # TODO(gaotingquan)
        # id_tensor = paddle.to_tensor(kernel_value, place=self.bn.weight.place)
        id_tensor = paddle.to_tensor(kernel_value)
        kernel = id_tensor
        _mean = self.bn._mean
        _variance = self.bn._variance
        gamma = self.bn.weight
        beta = self.bn.bias
        _epsilon = self.bn._epsilon
        std = (_variance + _epsilon).sqrt()
        t = (gamma / std).reshape([-1, 1, 1, 1])

        kernel_hat = kernel * t
        bias_hat = beta - _mean * gamma / std
        self.deploy_conv.weight.set_value(kernel_hat)
        self.deploy_conv.bias.set_value(bias_hat)
        self.deployed = True


class DiverseBranchBlock(RepBlock):
    def __init__(
            self,
            rep_blocks=[
                "ConvKxK", "ConvKxK", "ConvKxK", "ConvKxK", "ConvKxK",
                "ConvKxK", "ConvKxK"
            ],
            block_config,
            # in_channels,
            # out_channels,
            # kernel_size,
            # stride=1,
            # dilation=1,
            # groups=1,
            # internal_channels=None,  # internal channel between 1x1 and kxk
            # padding=None,
            # with_bn=True,
            act=None,
            ori_conv=None,
            recal_bn_fn=None):
        super().__init__()
        # TODO(gaotingquan): 
        # if not (out_channels == in_channels and stride == 1):
        #     branches[6] = 0
        self.deployed = False

        assert "ConvKxK" in rep_blocks
        self.block_config = block_config
        self.branch_list = []
        for block_name in rep_blocks:
            block = getattr(__name__, block_name)(block_config)
            self.branch_list.append(block)
        self.act = act

        if ori_conv is not None:
            self.recal_bn_fn = recal_bn_fn

        # self.active_branch_num = sum(branches)
        # self.branches = branches
        # self.in_channels = in_channels
        # self.kernel_size = kernel_size
        # self.out_channels = out_channels
        # self.stride = stride
        # self.dilation = dilation
        # self.groups = groups
        # self.padding = padding

        # if branches[0]:
        #     self.dbb_1x1 = ConvBN(in_channels=in_channels,
        #                           out_channels=out_channels,
        #                           kernel_size=1,
        #                           stride=stride,
        #                           padding=0,
        #                           groups=groups,
        #                           bn=bn)
        # if branches[1]:
        #     if internal_channels is None:
        #         internal_channels = in_channels
        #     self.dbb_1x1_kxk = nn.Sequential()
        #     self.dbb_1x1_kxk.add_sublayer(
        #         'conv1',
        #         nn.Conv2D(in_channels=in_channels,
        #                   out_channels=internal_channels,
        #                   kernel_size=1,
        #                   stride=1,
        #                   padding=0,
        #                   groups=groups,
        #                   bias_attr=False))
        #     self.dbb_1x1_kxk.add_sublayer(
        #         'bn1',
        #         BNAndPad(pad_pixels=padding,
        #                  num_features=internal_channels,
        #                  last_conv_bias=self.dbb_1x1_kxk.conv1.bias,
        #                  bn=bn))
        #     self.dbb_1x1_kxk.add_sublayer(
        #         'conv2',
        #         nn.Conv2D(in_channels=internal_channels,
        #                   out_channels=out_channels,
        #                   kernel_size=kernel_size,
        #                   stride=stride,
        #                   padding=0,
        #                   groups=groups,
        #                   bias_attr=False))
        #     self.dbb_1x1_kxk.add_sublayer(
        #         'bn2',
        #         bn(num_features=out_channels))
        # if branches[2]:
        #     self.dbb_1x1_avg = nn.Sequential()
        #     if self.groups < self.out_channels:
        #         self.dbb_1x1_avg.add_sublayer(
        #             'conv',
        #             nn.Conv2D(in_channels=in_channels,
        #                       out_channels=out_channels,
        #                       kernel_size=1,
        #                       stride=1,
        #                       padding=0,
        #                       groups=groups,
        #                       bias_attr=False))
        #         self.dbb_1x1_avg.add_sublayer(
        #             'bn',
        #             BNAndPad(pad_pixels=padding,
        #                      num_features=out_channels,
        #                      last_conv_bias=self.dbb_1x1_avg.conv.bias,
        #                      bn=bn))
        #         self.dbb_1x1_avg.add_sublayer(
        #             'avg',
        #             nn.AvgPool2D(kernel_size=kernel_size,
        #                          stride=stride,
        #                          padding=0))
        #     else:
        #         self.dbb_1x1_avg.add_sublayer(
        #             'avg',
        #             nn.AvgPool2D(kernel_size=kernel_size,
        #                          stride=stride,
        #                          padding=padding))
        #     self.dbb_1x1_avg.add_sublayer(
        #         'avgbn',
        #         bn(num_features=out_channels))
        # if branches[3]:
        #     self.dbb_kxk = ConvBN(in_channels=in_channels,
        #                           out_channels=out_channels,
        #                           kernel_size=kernel_size,
        #                           stride=stride,
        #                           padding=padding,
        #                           dilation=dilation,
        #                           groups=groups,
        #                           bias_attr=True,
        #                           bn=bn)
        # if branches[4]:
        #     self.dbb_1xk = ConvBN(in_channels=in_channels,
        #                           out_channels=out_channels,
        #                           kernel_size=(1, kernel_size),
        #                           stride=stride,
        #                           padding=(0, self.padding),
        #                           dilation=dilation,
        #                           groups=groups,
        #                           bias_attr=False,
        #                           bn=bn)
        # if branches[5]:
        #     self.dbb_kx1 = ConvBN(in_channels=in_channels,
        #                           out_channels=out_channels,
        #                           kernel_size=(kernel_size, 1),
        #                           stride=stride,
        #                           padding=(self.padding, 0),
        #                           dilation=dilation,
        #                           groups=groups,
        #                           bias_attr=False,
        #                           bn=bn)
        # if branches[6]:
        #     self.dbb_id = bn(num_features=out_channels)

    def branch_weights(self):
        def _cal_weight(data):
            return data.abs().mean().item()  # L1

        weights = [-1] * len(self.branches)
        kxk_weight = _cal_weight(self.dbb_kxk.bn.weight)
        # Make the weight of kxk branch as 1,
        # this is for better generalization of the thrd value (lambda)
        weights[3] = 1
        if self.branches[0]:
            weights[0] = _cal_weight(self.dbb_1x1.bn.weight) / kxk_weight
        if self.branches[1]:
            weights[1] = _cal_weight(self.dbb_1x1_kxk[-1].weight) / kxk_weight
        if self.branches[2]:
            weights[2] = _cal_weight(self.dbb_1x1_avg[-1].weight) / kxk_weight
        if self.branches[4]:
            weights[4] = _cal_weight(self.dbb_1xk.bn.weight) / kxk_weight
        if self.branches[5]:
            weights[5] = _cal_weight(self.dbb_kx1.bn.weight) / kxk_weight
        if self.branches[6]:
            weights[6] = _cal_weight(self.dbb_id.weight) / kxk_weight
        return weights

    def _reset_dbb(self,
                   kernel,
                   bias,
                   no_init_branches=[0, 0, 0, 0, 0, 0, 0, 0]):
        self._init_branch(self.dbb_kxk, set_zero=True, norm=1)
        if self.branches[0] and no_init_branches[0] == 0:
            self._init_branch(self.dbb_1x1)
        if self.branches[1] and no_init_branches[1] == 0:
            self._init_branch(self.dbb_1x1_kxk)
        if self.branches[2] and no_init_branches[2] == 0:
            self._init_branch(self.dbb_1x1_avg)
        if self.branches[4] and no_init_branches[4] == 0:
            self._init_branch(self.dbb_1xk)
        if self.branches[5] and no_init_branches[5] == 0:
            self._init_branch(self.dbb_kx1)
        if self.branches[6] and no_init_branches[6] == 0:
            self._init_branch(self.dbb_id)

        # TODO
        if self.recal_bn_fn is not None and sum(
                no_init_branches) == 0 and isinstance(kernel, nn.Parameter):
            self.dbb_kxk.conv.weight.data.copy_(kernel)
            if bias is not None:
                if self.dbbkxk.conv.bias is not None:
                    self.dbb_kxk.conv.bias.set_value(bias)
                # TODO: there may be a problem here.
                # because the bias is a tensor with stop_grad is True
                else:
                    self.dbb_kxk.conv.bias = bias
            self.recal_bn_fn(self)
            # TODO: paddle not support reset
            self.dbb_kxk.bn.reset_running_stats()
        cur_w, cur_b = self.get_actual_kernel(ignore_kxk=True)
        # reverse dbb transform
        new_w = paddle.to_tensor(kernel, place=cur_w.place) - cur_w
        if bias is not None:
            new_b = paddle.to_tensor(bias, place=cur_b.place) - cur_b
        else:
            new_b = -cur_b

        if isinstance(self.dbb_kxk.conv, nn.Conv2D):
            if isinstance(self.dbb_kxk.bn, nn.BatchNorm2D):
                self.dbb_kxk.bn.weight.set_value(
                    paddle.full_like(self.dbb_kxk.bn.weight))
                self.dbb_kxk.bn.bias.set_value(
                    paddle.zeros_like(self.dbb_kxk.bn.bias))
            self.dbb_kxk.conv.weight.set_value(new_w)
            # TODO: there may be a problem here.
            # because the bias is a tensor with stop_grad is True
            if self.dbb_kxk.conv.bias is not None:
                self.dbb_kxk.conv.bias.set_value(new_b)
            else:
                self.dbb_kxk.conv.bias = new_b
        elif isinstance(self.dbb_kxk.conv, DiverseBranchBlock):
            self.dbb_kxk.conv._reset_dbb(new_w, new_b)

    def _init_branch(self, branch, set_zero=False, norm=0.01):
        bns = []
        for layer in branch.layers():
            if isinstance(layer, nn.Conv2D):
                if set_zero:
                    layer.weight.set_value(paddle.zeros_like(layer.weight))
                else:
                    n = layer._kernel_size[0] * layer._kernel_size[
                        1] * layer._out_channels  # fan-out
                    layer.weight.set_value(
                        paddle.normal(
                            0, math.sqrt(2.0 / n), shape=layer.weight.shape))
                if layer.bias is not None:
                    layer.bias.set_value(paddle.zeros_like(layer.bias))
            elif isinstance(layer, nn.BatchNorm2D):
                bns.append(layer)
        for idx, layer in enumerate(bns):
            # TODO: paddle not support reset
            layer.reset_parameters()
            layer.reset_running_stats()
            if idx == len(bns) - 1:
                layer.weight.set_value(
                    paddle.full_like(norm))  # set to a small value
            else:
                layer.weight.set_value(paddle.ones_like(layer.weight))
            if layer.bias is not None:
                layer.bias.set_value(paddle.zeros_like(layer.bias))

    def get_actual_kernel(self, ignore_kxk=False):
        if self.deployed:
            return self.conv_deployed.weight, self.conv_deployed.bias
        ws = []
        bs = []
        if not ignore_kxk:  # kxk-bn
            if isinstance(self.dbb_kxk.conv, nn.Conv2D):
                w, b = self.dbb_kxk.conv.weight, self.dbb_kxk.conv.bias
            elif isinstance(self.dbb_kxk.conv, DiverseBranchBlock):
                w, b = self.dbb_kxk.conv.get_actual_kernel()
            if not isinstance(self.dbb_kxk.bn, nn.Identity):
                w, b = transI_fusebn(w, self.dbb_kxk.bn, b)
            ws.append(w.unsqueeze(0))
            bs.append(b.unsqueeze(0))
        if self.branches[0]:  # 1x1-bn
            w_1x1, b_1x1 = transI_fusebn(self.dbb_1x1.conv.weight,
                                         self.dbb_1x1.bn,
                                         self.dbb_1x1.conv.bias)
            w_1x1 = transVI_multiscale(w_1x1, self.kernel_size)
            ws.append(w_1x1.unsqueeze(0))
            bs.append(b_1x1.unsqueeze(0))
        if self.branches[1]:  # 1x1-bn-kxk-bn
            if isinstance(self.dbb_1x1_kxk.conv2, nn.Conv2D):
                w_1x1_kxk, b_1x1_kxk = transI_fusebn(
                    self.dbb_1x1_kxk.conv2.weight, self.dbb_1x1_kxk.bn2,
                    self.dbb_1x1_kxk.conv2.bias)
            elif isinstance(self.dbb_1x1_kxk.conv2, DiverseBranchBlock):
                w_1x1_kxk, b_1x1_kxk = \
                    self.dbb_1x1_kxk.conv2.get_actual_kernel()
                w_1x1_kxk, b_1x1_kxk = transI_fusebn(
                    w_1x1_kxk, self.dbb_1x1_kxk.bn2, b_1x1_kxk)
            w_1x1_kxk_first, b_1x1_kxk_first = transI_fusebn(
                self.dbb_1x1_kxk.conv1.weight, self.dbb_1x1_kxk.bn1,
                self.dbb_1x1_kxk.conv1.bias)
            w_1x1_kxk, b_1x1_kxk = transIII_1x1_kxk(
                w_1x1_kxk_first,
                b_1x1_kxk_first,
                w_1x1_kxk,
                b_1x1_kxk,
                groups=self.groups)
            ws.append(w_1x1_kxk.unsqueeze(0))
            bs.append(b_1x1_kxk.unsqueeze(0))
        if self.branches[2]:  # 1x1-bn-avg-bn
            w_1x1_avg = transV_avg(self.out_channels, self.kernel_size,
                                   self.groups)
            w_1x1_avg, b_1x1_avg = transI_fusebn(
                paddle.to_tensor(
                    w_1x1_avg, place=self.dbb_1x1_avg.avgbn.weight.place),
                self.dbb_1x1_avg.avgbn,
                None)
            if self.groups < self.out_channels:
                w_1x1_avg_first, b_1x1_avg_first = transI_fusebn(
                    self.dbb_1x1_avg.conv.weight, self.dbb_1x1_avg.bn,
                    self.dbb_1x1_avg.conv.bias)
                w_1x1_avg, b_1x1_avg = transIII_1x1_kxk(
                    w_1x1_avg_first,
                    b_1x1_avg_first,
                    w_1x1_avg,
                    b_1x1_avg,
                    groups=self.groups)
            ws.append(w_1x1_avg.unsqueeze(0))
            bs.append(b_1x1_avg.unsqueeze(0))
        if self.branches[4]:  # 1xk-bn
            w_1xk, b_1xk = transI_fusebn(self.dbb_1xk.conv.weight,
                                         self.dbb_1xk.bn,
                                         self.dbb_1xk.conv.bias)
            w_1xk = transVI_multiscale(w_1xk, self.kernel_size)
            ws.append(w_1xk.unsqueeze(0))
            bs.append(b_1xk.unsqueeze(0))
        if self.branches[5]:  # kx1-bn
            w_kx1, b_kx1 = transI_fusebn(self.dbb_kx1.conv.weight,
                                         self.dbb_kx1.bn,
                                         self.dbb_kx1.conv.bias)
            w_kx1 = transVI_multiscale(w_kx1, self.kernel_size)
            ws.append(w_kx1.unsqueeze(0))
            bs.append(b_kx1.unsqueeze(0))
        if self.branches[6]:  # BN
            w_id, b_id = transIX_bn_to_1x1(self.dbb_id,
                                           self.dbb_kxk.conv.in_channels,
                                           self.dbb_kxk.conv.groups)
            w_id = transVI_multiscale(w_id, self.kernel_size)
            ws.append(w_id.unsqueeze(0))
            bs.append(b_id.unsqueeze(0))

        ws = paddle.concat(ws)
        bs = paddle.concat(bs)

        return transII_addbranch(ws, bs)

    # def switch_to_deploy(self):
    def re_parameterize(self):
        if self.deployed:
            return
        w, b = self.get_actual_kernel()

        self.conv_deployed = nn.Conv2D(
            in_channels=self.block_config["in_channels"],
            out_channels=self.block_config["out_channels"],
            kernel_size=self.block_config["kernel_size"],
            stride=self.block_config["stride"],
            padding=self.block_config["padding"],
            dilation=self.block_config["dilation"],
            groups=self.block_config["groups"],
            bias_attr=True)

        self.conv_deployed.weight.set_value(w)
        self.conv_deployed.bias.set_value(b)
        # TODO(gaotingquan): is OK ?
        # self.branch_list = []
        del self.branch_list
        self.deployed = True
        # TODO: paddle not support detach_()
        # for para in self.parameters():
        #     para.detach_()
        # if self.branches[0]:
        #     self.__delattr__('dbb_1x1')
        # if self.branches[1]:
        #     self.__delattr__('dbb_1x1_kxk')
        # if self.branches[2]:
        #     self.__delattr__('dbb_1x1_avg')
        # if self.branches[3]:
        #     self.__delattr__('dbb_kxk')
        # if self.branches[4]:
        #     self.__delattr__('dbb_1xk')
        # if self.branches[5]:
        #     self.__delattr__('dbb_kx1')
        # if self.branches[6]:
        #     self.__delattr__('dbb_id')

    def forward(self, x):
        if self.deployed:
            return self.act(self.conv_deployed(x))

        out = 0
        for branch in self.branch_list:
            out += branch(x)
        return self.act(out)

    def cut_branch(self, branches):
        ori_w, ori_b = self.get_actual_kernel()
        _branch_names = [
            'dbb_1x1', 'dbb_1x1_kxk', 'dbb_1x1_avg', 'dbb_kxk', 'dbb_1xk',
            'dbb_kx1', 'dbb_id'
        ]
        for idx, status in enumerate(branches):
            if status == 0 and self.branches[idx] == 1:
                self.branches[idx] = 0
                self.__delattr__(_branch_names[idx])
        self._reset_dbb(ori_w, ori_b, no_init_branches=branches)
