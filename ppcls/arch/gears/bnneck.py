import paddle


class BNNeck(paddle.nn.Layer):
    def __init__(self, num_filters, trainable=False):
        super(BNNeck, self).__init__()
        self.num_filters = num_filters

        self.bn = paddle.nn.BatchNorm(
            self.num_filters)
        if not trainable:
            self.bn.weight.trainable = False
            self.bn.bias.trainable = False

    def forward(self, input, label=None):
        out = self.bn(input)
        return out
