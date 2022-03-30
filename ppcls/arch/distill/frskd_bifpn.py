import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ['BiFPNc', 'BiFPN']


class BiFPNc(nn.Layer):
    def __init__(self,
                 network_channel=[64, 128, 256, 512],
                 num_classes=1000,
                 repeat=1,
                 depth=2,
                 width=2,
                 num_features=4,
                 preact=False,
                 keys=["blocks[1]", "blocks[3]", "blocks[5]", "blocks[7]"]):
        super(BiFPNc, self).__init__()
        repeat = repeat
        depth = [depth] * 3
        width = width

        self.num_features = num_features
        self.layers = nn.LayerList()
        self.preact = preact
        self.keys = keys

        self.net_channels = [x * width for x in network_channel]
        for i in range(repeat):
            self.layers.append(
                BiFPNc_layer(i == 0, DepthConvBlock, network_channel, depth,
                             width))

        self.fc = nn.Linear(self.net_channels[-1], num_classes)

    def forward(self, features_dict):
        feats = [features_dict[key] for key in self.keys]
        feats = feats[-self.num_features:]

        for i in range(len(self.layers)):
            layer_preact = self.preact and i == len(self.layers) - 1
            feats = self.layers[i](feats, layer_preact)

        out = F.adaptive_avg_pool2d(F.relu(feats[-1]),
                                    (1, 1))  # for preact case
        out = out.reshape([out.shape[0], -1])
        out = self.fc(out)
        out_dict = dict()
        out_dict["logits_fpn"] = out
        for i in range(len(feats)):
            out_dict["blocks[{}]_fpn".format(i)] = feats[i]
        return out_dict

    def get_bn_before_relu(self):
        layer = self.layers[-1]
        bn = [layer.up_conv[0].conv[-1][-1]]
        for down_conv in layer.down_conv:
            bn.append(down_conv.conv[-1][-1])
        return bn


class BiFPN(nn.Layer):
    def __init__(self,
                 network_channel=[64, 128, 256, 512],
                 num_classes=1000,
                 repeat=1,
                 depth=2,
                 num_channels=256,
                 num_features=4,
                 preact=False,
                 keys=["blocks[1]", "blocks[3]", "blocks[5]", "blocks[7]"]):
        super(BiFPN, self).__init__()
        repeat = repeat
        num_channels = num_channels
        depth = [depth] * 3

        self.num_features = num_features
        self.layers = nn.LayerList()
        self.preact = preact
        self.keys = keys

        self.net_channels = [num_channels] * len(network_channel)
        for i in range(repeat):
            self.layers.append(
                BiFPN_layer(i == 0, DepthConvBlock, network_channel,
                            num_channels, depth))

        self.linear = nn.Linear(num_channels, num_classes)

    def forward(self, feats):
        feats = [feats[key] for key in self.keys]
        feats = feats[-self.num_features:]

        for i in range(len(self.layers)):
            layer_preact = self.preact and i == len(self.layers) - 1
            feats = self.layers[i](feats, layer_preact)

        out = F.adaptive_avg_pool2d(F.relu(feats[-1]),
                                    (1, 1))  # for preact case
        out = out.reshape([out.shape[0], -1])
        out = self.linear(out)
        return feats, out

    def get_bn_before_relu(self):
        layer = self.layers[-1]
        bn = [layer.up_conv[0].conv[-1][-1]]
        for down_conv in layer.down_conv:
            bn.append(down_conv.conv[-1][-1])
        return bn


class BiFPNc_layer(nn.Layer):
    def __init__(self, first_time, block, network_channel, depth, width):
        super(BiFPNc_layer, self).__init__()
        lat_depth, up_depth, down_depth = depth
        self.first_time = first_time

        self.lat_conv = nn.LayerList()
        self.lat_conv2 = nn.LayerList()

        self.up_conv = nn.LayerList()
        self.up_weight = nn.ParameterList()

        self.down_conv = nn.LayerList()
        self.down_weight = nn.ParameterList()
        self.down_sample = nn.LayerList()
        self.up_sample = nn.LayerList()

        for i, channels in enumerate(network_channel):
            if self.first_time:
                self.lat_conv.append(
                    block(channels, channels * width, 1, 1, 0, lat_depth))

            if i != 0:
                self.lat_conv2.append(
                    block(channels, channels * width, 1, 1, 0, lat_depth))
                self.down_conv.append(
                    block(channels * width, channels * width, 3, 1, 1,
                          down_depth))
                num_input = 3 if i < len(network_channel) - 1 else 2

                self.down_weight.append(
                    self.create_parameter(
                        shape=[num_input],
                        dtype=paddle.float32,
                        default_initializer=nn.initializer.Constant(
                            value=1.0)))
                self.down_sample.append(
                    nn.Sequential(
                        MaxPool2dStaticSamePadding(3, 2),
                        block(network_channel[i - 1] * width, channels * width,
                              1, 1, 0, 1)))

            if i != len(network_channel) - 1:
                self.up_sample.append(
                    nn.Sequential(
                        nn.Upsample(
                            scale_factor=2, mode='nearest'),
                        block(network_channel[i + 1] * width, channels * width,
                              1, 1, 0, 1)))
                self.up_conv.append(
                    block(channels * width, channels * width, 3, 1, 1,
                          up_depth))
                self.up_weight.append(
                    self.create_parameter(
                        shape=[2],
                        dtype=paddle.float32,
                        default_initializer=nn.initializer.Constant(
                            value=1.0)))

        self.relu = nn.ReLU()

        self.epsilon = 1e-6

    def forward(self, inputs, preact=False):
        input_trans = [
            self.lat_conv2[i - 1](F.relu(inputs[i]))
            for i in range(1, len(inputs))
        ]
        if self.first_time:
            inputs = [
                self.lat_conv[i](F.relu(inputs[i]))
                for i in range(len(inputs))
            ]  # for od case

        # up
        up_sample = [inputs[-1]]
        out_layer = []
        for i in range(1, len(inputs)):
            w = self.relu(self.up_weight[len(self.up_weight) - i])
            w = w / (paddle.sum(w, axis=0) + self.epsilon)

            up_sample.insert(0, self.up_conv[-i](w[0] * F.relu(inputs[
                -i - 1]) + w[1] * self.up_sample[-i](F.relu(up_sample[0]))))

        out_layer.append(up_sample[0])

        # down
        for i in range(1, len(inputs)):
            w = self.relu(self.down_weight[i - 1])
            w = w / (paddle.sum(w, axis=0) + self.epsilon)
            if i < len(inputs) - 1:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(
                    input_trans[i - 1]) + w[1] * F.relu(up_sample[i]) + w[
                        2] * self.down_sample[i - 1](F.relu(out_layer[-1]))))
            else:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(
                    input_trans[i - 1]) + w[1] * self.down_sample[i - 1](
                        F.relu(out_layer[-1]))))

        if not preact:
            return [F.relu(f) for f in out_layer]
        return out_layer


class BiFPN_layer(nn.Layer):
    def __init__(self, first_time, block, network_channel, num_channels,
                 depth):
        super(BiFPN_layer, self).__init__()
        lat_depth, up_depth, down_depth = depth
        self.first_time = first_time

        self.lat_conv = nn.LayerList()
        self.lat_conv2 = nn.LayerList()

        self.up_conv = nn.LayerList()
        self.up_weight = nn.ParameterList()

        self.down_conv = nn.LayerList()
        self.down_weight = nn.ParameterList()
        self.down_sample = nn.LayerList()

        for i, channels in enumerate(network_channel):
            if self.first_time:
                self.lat_conv.append(
                    block(channels, num_channels, 1, 1, 0, lat_depth))

            if i != 0:
                self.lat_conv2.append(
                    block(channels if self.first_time else num_channels,
                          num_channels, 1, 1, 0, lat_depth))
                self.down_conv.append(
                    block(num_channels, num_channels, 3, 1, 1, down_depth))
                num_input = 3 if i < len(network_channel) - 1 else 2
                self.down_weight.append(
                    nn.Parameter(
                        paddle.ones(
                            num_input, dtype=paddle.float32),
                        requires_grad=True))
                self.down_sample.append(MaxPool2dStaticSamePadding(3, 2))

            if i != len(network_channel) - 1:
                self.up_conv.append(
                    block(num_channels, num_channels, 3, 1, 1, up_depth))
                self.up_weight.append(
                    nn.Parameter(
                        paddle.ones(
                            2, dtype=paddle.float32),
                        requires_grad=True))

        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.relu = nn.ReLU()

        self.epsilon = 1e-6

    def forward(self, inputs, preact=False):
        input_trans = [
            self.lat_conv2[i - 1](F.relu(inputs[i]))
            for i in range(1, len(inputs))
        ]
        if self.first_time:
            inputs = [
                self.lat_conv[i](F.relu(inputs[i]))
                for i in range(len(inputs))
            ]  # for od case

        # up
        up_sample = [inputs[-1]]
        out_layer = []
        for i in range(1, len(inputs)):
            w = self.relu(self.up_weight[-i])
            w = w / (paddle.sum(w, axis=0) + self.epsilon)
            up_sample.insert(0, self.up_conv[-i](w[0] * F.relu(inputs[
                -i - 1]) + w[1] * self.up_sample(F.relu(up_sample[0]))))
        out_layer.append(up_sample[0])

        # down
        for i in range(1, len(inputs)):
            w = self.relu(self.down_weight[i - 1])
            w = w / (paddle.sum(w, axis=0) + self.epsilon)
            if i < len(inputs) - 1:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(
                    input_trans[i - 1]) + w[1] * F.relu(up_sample[i]) + w[
                        2] * self.down_sample[i - 1](F.relu(out_layer[-1]))))
            else:
                out_layer.append(self.down_conv[i - 1](w[0] * F.relu(
                    input_trans[i - 1]) + w[1] * self.down_sample[i - 1](
                        F.relu(out_layer[-1]))))

        if not preact:
            return [F.relu(f) for f in out_layer]
        return out_layer


class MaxPool2dStaticSamePadding(nn.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2D(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.ksize

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride[1]) - 1
                   ) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1
                   ) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x


class DepthConvBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 depth=1):
        super(DepthConvBlock, self).__init__()
        conv = []
        if kernel_size == 1:
            conv.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias_attr=False),
                    nn.BatchNorm2D(out_channels),
                    nn.ReLU(), ))
        else:
            conv.append(
                nn.Sequential(
                    nn.Conv2D(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        bias_attr=False,
                        groups=in_channels),
                    nn.Conv2D(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias_attr=False),
                    nn.BatchNorm2D(out_channels), ))
            for i in range(depth - 1):
                conv.append(
                    nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2D(
                            out_channels,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias_attr=False,
                            groups=out_channels),
                        nn.Conv2D(
                            out_channels,
                            out_channels,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias_attr=False),
                        nn.BatchNorm2D(out_channels), ))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)
