import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.utils import logger


class ClipLoss(nn.Layer):
    """
    ClipLoss
    """

    def __init__(self, reduction="mean", epsilon=None):
        super().__init__()

    def forward(self, logits_per_image, logits_per_text, labels):
        total_loss = (F.cross_entropy(logits_per_image, labels) +
                      F.cross_entropy(logits_per_text, labels)) / 2

        return {"contrastive_loss": total_loss}
