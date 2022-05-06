# This code is based on AdaFace(https://github.com/mk-minchul/AdaFace)
from paddle.nn import Layer
import math
import paddle


def l2_norm(input, axis=1):
    norm = paddle.norm(input, 2, axis, True)
    output = paddle.divide(input, norm)
    return output


class AdaMargin(Layer):
    def __init__(
            self,
            embedding_size=512,
            class_num=70722,
            m=0.4,
            h=0.333,
            s=64.,
            t_alpha=1.0, ):
        super(AdaMargin, self).__init__()
        self.classnum = class_num
        self.kernel = self.create_parameter(
            [embedding_size, class_num], attr=paddle.nn.initializer.Uniform())

        # initial kernel
        # self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', paddle.zeros([1]), persistable=True)
        self.register_buffer(
            'batch_mean', paddle.ones([1]) * 20, persistable=True)
        self.register_buffer(
            'batch_std', paddle.ones([1]) * 100, persistable=True)

        print('\n\AdaFace with the following property')
        print('self.m', self.m)
        print('self.h', self.h)
        print('self.s', self.s)
        print('self.t_alpha', self.t_alpha)

    def forward(self, embbedings, norms, label):

        kernel_norm = l2_norm(self.kernel, axis=0)
        cosine = paddle.mm(embbedings, kernel_norm)
        cosine = paddle.clip(cosine, -1 + self.eps,
                             1 - self.eps)  # for stability

        safe_norms = paddle.clip(norms, min=0.001, max=100)  # for stability
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with paddle.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha
                                                     ) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha
                                                   ) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (
            self.batch_std + self.eps)  # 66% between -1, 1
        margin_scaler = margin_scaler * self.h  # 68% between -0.333 ,0.333 when h:0.333
        margin_scaler = paddle.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = paddle.nn.functional.one_hot(label, self.classnum)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = paddle.acos(cosine)
        theta_m = paddle.clip(
            theta + m_arc, min=self.eps, max=math.pi - self.eps)
        cosine = paddle.cos(theta_m)

        # g_additive
        m_cos = paddle.nn.functional.one_hot(label, self.classnum)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m
