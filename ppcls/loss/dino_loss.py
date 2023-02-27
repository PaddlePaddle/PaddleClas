import numpy as np
from paddle import nn
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist


class DINOLoss(nn.Layer):

    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", paddle.zeros((1, out_dim)))

        # apply a warm up for the teacher temperature because
        # a too high temperature make the training instable in the beginning
        warm_up_teacher_sch = np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs)
        teacher_sch = np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        self.teacher_temp_schedule = np.concatenate([warm_up_teacher_sch, teacher_sch]).astype("float32")

    def forward(self, student_output: paddle.Tensor, teacher_output: paddle.Tensor, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, axis=-1)
        teacher_out = teacher_out.detach().chunk(2)

        # compute cross-entropy loss
        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # skip cases where student and teacher operate on the same view
                    continue
                loss = paddle.sum(-q * F.log_softmax(student_out[v], axis=-1), axis=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @paddle.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = paddle.sum(teacher_output, axis=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
