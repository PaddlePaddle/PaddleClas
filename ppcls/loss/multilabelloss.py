import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def ratio2weight(targets, ratio):
    pos_weights = targets * (1. - ratio)
    neg_weights = (1. - targets) * ratio
    weights = paddle.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1).astype(weights.dtype)

    return weights


class MultiLabelLoss(nn.Layer):
    """
    Multi-label loss
    """

    def __init__(self, epsilon=None, size_sum=False, weight_ratio=False):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        self.weight_ratio = weight_ratio
        self.size_sum = size_sum

    def _labelsmoothing(self, target, class_num):
        if target.ndim == 1 or target.shape[-1] != class_num:
            one_hot_target = F.one_hot(target, class_num)
        else:
            one_hot_target = target
        soft_target = F.label_smooth(one_hot_target, epsilon=self.epsilon)
        soft_target = paddle.reshape(soft_target, shape=[-1, class_num])
        return soft_target

    def _binary_crossentropy(self, input, target, class_num):
        if self.weight_ratio:
            target, label_ratio = target[:, 0, :], target[:, 1, :]
        else:
            target = target[:, 0, :]
        if self.epsilon is not None:
            target = self._labelsmoothing(target, class_num)
        cost = F.binary_cross_entropy_with_logits(
            logit=input, label=target, reduction='none')

        if self.weight_ratio:
            targets_mask = paddle.cast(target > 0.5, 'float32')
            weight = ratio2weight(targets_mask, paddle.to_tensor(label_ratio))
            weight = weight * (target > -1).astype(weight.dtype)
            cost = cost * weight

        if self.size_sum:
            cost = cost.sum(1).mean() if self.size_sum else cost.mean()

        return cost

    def forward(self, x, target):
        if isinstance(x, dict):
            x = x["logits"]
        class_num = x.shape[-1]
        loss = self._binary_crossentropy(x, target, class_num)
        loss = loss.mean()
        return {"MultiLabelLoss": loss}


class MultiLabelAsymmetricLoss(nn.Layer):
    """
    Multi-label asymmetric loss, introduced by
    Emanuel Ben-Baruch at el. in https://arxiv.org/pdf/2009.14119v4.pdf.
    """

    def __init__(self,
                 gamma_pos=1,
                 gamma_neg=4,
                 clip=0.05,
                 epsilon=1e-8,
                 disable_focal_loss_grad=True,
                 reduction="sum"):
        super().__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.epsilon = epsilon
        self.disable_focal_loss_grad = disable_focal_loss_grad
        assert reduction in ["mean", "sum", "none"]
        self.reduction = reduction

    def forward(self, x, target):
        if isinstance(x, dict):
            x = x["logits"]
        pred_sigmoid = F.sigmoid(x)
        target = target.astype(pred_sigmoid.dtype)

        # Asymmetric Clipping and Basic CE calculation
        if self.clip and self.clip > 0:
            pt = (1 - pred_sigmoid + self.clip).clip(max=1) \
                * (1 - target) + pred_sigmoid * target
        else:
            pt = (1 - pred_sigmoid) * (1 - target) + pred_sigmoid * target

        # Asymmetric Focusing
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(False)
        asymmetric_weight = (
            1 - pt
        ).pow(self.gamma_pos * target + self.gamma_neg * (1 - target))
        if self.disable_focal_loss_grad:
            paddle.set_grad_enabled(True)

        loss = -paddle.log(pt.clip(min=self.epsilon)) * asymmetric_weight

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return {"MultiLabelAsymmetricLoss": loss}
