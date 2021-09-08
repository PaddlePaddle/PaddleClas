from paddle import nn
import paddle

class IdentityHead(nn.Layer):
    def __init__(self, binarize_method = "none", embedding_size = 256):
        super(IdentityHead, self).__init__()
        self.binarize_method = binarize_method
        self.embedding_size = embedding_size
        self.multiplier = self._init_multiplier(embedding_size)

    def forward(self, x, label=None):
        if self.binarize_method == "round":
            x = paddle.round(x)

        if self.binarize_method == "sign":
            x = (paddle.sign(x) + 1.0) / 2.0

        if self.binarize_method == "round" or self.binarize_method == "sign":
            x = self._binary_to_byte(x, self.multiplier)

        return {"features": x, "logits": None}

    def _init_multiplier(self, embedding_size):
        unit = paddle.to_tensor([128, 64, 32, 16, 8, 4, 2, 1])
        repeat = embedding_size // 8
        assert embedding_size % 8 == 0, "The binary index only support vectors with sizes multiple of 8"
        unit = paddle.broadcast_to(unit, shape=[repeat, 8])
        multiplier = paddle.reshape(unit, shape=[1, -1]).astype("float32")
        return multiplier

    def _binary_to_byte(self, input_tensor, multiplier):
        tmp = paddle.multiply(input_tensor, multiplier)
        tmp = paddle.reshape(tmp, shape=[tmp.shape[0], -1, 8])
        byte = paddle.sum(tmp, axis=-1).astype("uint8")
        return byte