import paddle
import paddle.nn as nn
import paddle.nn.functional as F


def ratio2weight_1(targets, ratio):
    '''
    Math formula:
    ```
    w_j = y_{ij} * e^{1 - r_j} + (1 - {y_ij}) * e^{r_j}
    ```
    REF: https://arxiv.org/abs/2107.03576v2
    '''

    pos_weights = targets * (1. - ratio)
    neg_weights = (1. - targets) * ratio
    weights = paddle.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1)

    return weights


def ratio2weight_2(targets, ratio):
    '''
    Math formula:
    ```
    w_j = y_{ij} * \sqrt{\frac{1}{2 * r_j}} + (1 - {y_ij}) * \sqrt{\frac{1}{2 * (1 - r_j)}}
    ```
    REF: https://arxiv.org/abs/2107.03576v2
    '''

    pos_weights = targets * ratio
    neg_weights = (1. - targets) * (1 - ratio)
    weights = paddle.sqrt(0.5 * paddle.reciprocal(neg_weights + pos_weights))

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1)

    return weights


def ratio2weight_3(targets, ratio, alpha):
    '''
    Math formula:
    ```
    w_j = y_{ij} * \frac{(1/r_j)^\alpha}{(1/r_j)^\alpha + (1/(1 - r_j))^\alpha} 
          + (1 - {y_ij}) * \frac{(1/(1 - r_j))^\alpha}{(1/r_j)^\alpha + (1/(1 - r_j))^\alpha} 
    ```
    REF: https://arxiv.org/abs/2107.03576v2
    '''

    pos_weights = targets * ratio
    neg_weights = (1. - targets) * (1 - ratio)
    combined_weights = pos_weights + neg_weights
    weights = paddle.divide(
        paddle.reciprocal(paddle.pow(combined_weights, alpha)),
        paddle.reciprocal(paddle.pow(combined_weights, alpha)) +
        paddle.reciprocal((paddle.pow(1 - combined_weights, alpha)))
    )

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights = weights - weights * (targets > 1)

    return weights


class MultiLabelLoss(nn.Layer):
    """
    Multi-label loss
    """

    def __init__(self, epsilon=None, size_sum=False, weight_type=1, weight_ratio=False, weight_alpha=False):
        super().__init__()
        if epsilon is not None and (epsilon <= 0 or epsilon >= 1):
            epsilon = None
        self.epsilon = epsilon
        self.weight_type = weight_type
        self.weight_alpha = weight_alpha
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
        if self.epsilon is not None:
            target = self._labelsmoothing(target, class_num)
        cost = F.binary_cross_entropy_with_logits(
            logit=input, label=target, reduction='none')

        if self.weight_ratio:
            targets_mask = paddle.cast(target > 0.5, 'float32')
            if self.weight_type == 2:
                weight = ratio2weight_2(
                    targets_mask, paddle.to_tensor(label_ratio))
            elif self.weight_type == 3:
                weight = ratio2weight_3(targets_mask, paddle.to_tensor(
                    label_ratio), self.weight_alpha)
            else:
                weight = ratio2weight_1(
                    targets_mask, paddle.to_tensor(label_ratio))
            weight = weight * (target > -1)
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
