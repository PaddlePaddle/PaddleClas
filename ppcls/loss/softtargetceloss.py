import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class SoftTargetCrossEntropy(nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        loss = paddle.sum(-target * F.log_softmax(x, axis=-1), axis=-1)
        loss = loss.mean()
        return {"SoftTargetCELoss": loss}

    def __str__(self, ):
        return type(self).__name__
