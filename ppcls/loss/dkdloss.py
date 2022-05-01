import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class DKDLoss(nn.Layer):
    """
    DKDLoss
    """

    def __init__(self,
                 model_name_pairs=[],
                 key=None,
                 temperature=1.0,
                 alpha=1.0,
                 beta=1.0,
                 name="loss_dkd"):
        super().__init__()
        self.key = key
        self.model_name_pairs = model_name_pairs
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.name = name

    def forward(self, predicts, batch):
        loss_dict = dict()
        for idx, pair in enumerate(self.model_name_pairs):
            out1 = predicts[pair[0]]
            out2 = predicts[pair[1]]
            if self.key is not None:
                out1 = out1[self.key]
                out2 = out2[self.key]
            loss = dkd_loss(out1, out2, batch, self.alpha, self.beta,
                            self.temperature)
            loss_dict[self.name] = loss
        return loss_dict


def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = 1 - gt_mask
    pred_student = F.softmax(logits_student / temperature, axis=1)
    pred_teacher = F.softmax(logits_teacher / temperature, axis=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)
    log_pred_student = paddle.log(pred_student)
    tckd_loss = (F.kl_div(
        log_pred_student, pred_teacher,
        reduction='sum') * (temperature**2) / target.shape[0])
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, axis=1)
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, axis=1)
    nckd_loss = (F.kl_div(
        log_pred_student_part2, pred_teacher_part2,
        reduction='sum') * (temperature**2) / target.shape[0])
    return alpha * tckd_loss + beta * nckd_loss


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
