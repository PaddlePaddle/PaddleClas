import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DKDLoss(nn.Layer):
    """
    DKDLoss
    Reference: https://arxiv.org/abs/2203.08679
    Code was heavily based on https://github.com/megvii-research/mdistiller
    """

    def __init__(self,
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0,
                 use_target_as_gt=False):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.use_target_as_gt = use_target_as_gt

    def forward(self, logits_student, logits_teacher, target=None):
        if target is None or self.use_target_as_gt:
            target = logits_teacher.argmax(axis=-1)
        gt_mask = _get_gt_mask(logits_student, target)
        other_mask = 1 - gt_mask
        pred_student = F.softmax(logits_student / self.temperature, axis=1)
        pred_teacher = F.softmax(logits_teacher / self.temperature, axis=1)
        pred_student = cat_mask(pred_student, gt_mask, other_mask)
        pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
        log_pred_student = paddle.log(pred_student)
        tckd_loss = (F.kl_div(
            log_pred_student, pred_teacher,
            reduction='sum') * (self.temperature**2) / target.shape[0])
        pred_teacher_part2 = F.softmax(
            logits_teacher / self.temperature - 1000.0 * gt_mask, axis=1)
        log_pred_student_part2 = F.log_softmax(
            logits_student / self.temperature - 1000.0 * gt_mask, axis=1)
        nckd_loss = (F.kl_div(
            log_pred_student_part2, pred_teacher_part2,
            reduction='sum') * (self.temperature**2) / target.shape[0])
        return self.alpha * tckd_loss + self.beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape([-1]).unsqueeze(1)
    updates = paddle.ones_like(target)
    mask = scatter(
        paddle.zeros_like(logits), target, updates.astype('float32'))
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(axis=1, keepdim=True)
    t2 = (t * mask2).sum(axis=1, keepdim=True)
    rt = paddle.concat([t1, t2], axis=1)
    return rt


def scatter(x, index, updates):
    i, j = index.shape
    grid_x, grid_y = paddle.meshgrid(paddle.arange(i), paddle.arange(j))
    index = paddle.stack([grid_x.flatten(), index.flatten()], axis=1)
    updates_index = paddle.stack([grid_x.flatten(), grid_y.flatten()], axis=1)
    updates = paddle.gather_nd(updates, index=updates_index)
    return paddle.scatter_nd_add(x, index, updates)
