# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# reference: https://arxiv.org/abs/1611.05431

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.nn as nn
from paddle.vision.models import ResNet
from paddle.vision.models.resnet import BottleneckBlock
# do not using paddleclas model resnet50
# from backbone.legendary_models import *
from ppcls.arch.init_weight import kaiming_init, constant_init, normal_init

# TODO NO UPLOAD
MODEL_URLS = {"MoCoV1": "UNKNOWN", "MoCoV2": "UNKNOWN"}

__all__ = list(MODEL_URLS.keys())


class LinearNeck(nn.Layer):
    """Linear neck: fc only.
    """

    def __init__(self, in_channels, out_channels, with_avg_pool=False):
        super(LinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(in_channels, out_channels)

    def forward(self, x):

        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.fc(x.reshape([x.shape[0], -1]))


class NonLinearNeck(nn.Layer):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=False):
        super(NonLinearNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2D((1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(), nn.Linear(hid_channels, out_channels))

    def forward(self, x):

        if self.with_avg_pool:
            x = self.avgpool(x)
        return self.mlp(x.reshape([x.shape[0], -1]))


class ContrastiveHead(nn.Layer):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1, **kwargs):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature

    def forward(self, pos, neg):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.shape[0]
        logits = paddle.concat((pos, neg), axis=1)
        logits /= self.temperature
        labels = paddle.zeros((N, ), dtype='int64')
        outputs = dict()
        outputs['loss'] = self.criterion(logits, labels)
        return outputs


class MoCo(nn.Layer):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, arch_config, dim=128, K=65536, m=0.999, T=0.07):
        """
        initialize `MoCoV1` or `MoCoV2` model depends on args
        Args:
            arch_config (dict): config of backbone(ResNet50), neck and head.
            scale (list|tuple): Range of size of the origin size cropped. Default: (0.08, 1.0)
            dim (int): feature dimension. Default: 128.
            K (int): queue size; number of negative keys. Default: 65536.
            m (float): moco momentum of updating key encoder. Default: 0.999.
            T (float): softmax temperature. Default: 0.07.
        """
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T

        # build backbone
        backbone_config = arch_config['backbone']
        neck_config = arch_config['neck']
        head_config = arch_config['head']

        # build neck
        backbone_name = backbone_config.pop('name')
        # neck
        neck_name = neck_config.pop('name')
        head_name = head_config.pop('name')
        # instantlize head net
        self.head = eval(head_name)(**head_config)
        """ 
        # create the encoders 
        # return layers defore reset50 avg_pool, include avg_pool layer
        self.encoder_q = nn.Sequential(
        backbone(**backbone_config).stop_after(stop_layer_name='avg_pool'),
        nn.Flatten(), 
        neck(**neck_config)
        ) 

        self.encoder_k = nn.Sequential(
        backbone(**backbone_config).stop_after(stop_layer_name='avg_pool'),
        nn.Flatten(),  
        neck(**neck_config)
        )
        r_50 = ResNet(BottleneckBlock, 50, num_classes=0, with_pool=False)
        """

        self.encoder_q = nn.Sequential(
            eval(backbone_name)(BottleneckBlock,
                                num_classes=0,
                                with_pool=False,
                                **backbone_config),
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Flatten(),
            eval(neck_name)(**neck_config))

        self.encoder_k = nn.Sequential(
            eval(backbone_name)(BottleneckBlock,
                                num_classes=0,
                                with_pool=False,
                                **backbone_config),
            nn.AdaptiveAvgPool2D((1, 1)),
            nn.Flatten(),
            eval(neck_name)(**neck_config))

        self.backbone = self.encoder_q[0]

        # initialize every layer with kaiming
        self.init_parameters()

        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.set_value(param_q)  # moco initialize
            param_k.stop_gradient = True  # not update by gradient

        # frozen bn normal
        freeze_batchnorm_statictis(self.encoder_k)

        # create the queue
        self.register_buffer("queue", paddle.randn([dim, K]))
        self.queue = nn.functional.normalize(self.queue, axis=0)

        self.register_buffer("queue_ptr", paddle.zeros([1], 'int64'))

    @paddle.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            paddle.assign((param_k * self.m + param_q * (1. - self.m)),
                          param_k)
            param_k.stop_gradient = True

    @paddle.no_grad()
    def _dequeue_and_enqueue(self, keys):
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr[0])
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose([1, 0])
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @paddle.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = paddle.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = paddle.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_shuffle.reshape([num_gpus, -1])[gpu_idx]
        return paddle.index_select(x_gather, idx_this), idx_unshuffle

    @paddle.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = paddle.distributed.get_rank()
        idx_this = idx_unshuffle.reshape([num_gpus, -1])[gpu_idx]

        return paddle.index_select(x_gather, idx_this)

    def train_iter(self, *inputs, **kwargs):
        img_q, img_k = inputs

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = nn.functional.normalize(q, axis=1)

        # compute key features
        with paddle.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            img_k = paddle.to_tensor(img_k)
            im_k, idx_unshuffle = self._batch_shuffle_ddp(img_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, axis=1)

            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # FIXME: Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = paddle.sum(q * k, axis=1).unsqueeze(-1)
        # negative logits: NxK
        l_neg = paddle.matmul(q, self.queue.clone().detach())

        outputs = self.head(l_pos, l_neg)
        self._dequeue_and_enqueue(k)
        # add return label

        return outputs

    def forward(self, *inputs, mode='train', **kwargs):
        if mode == 'train':
            return self.train_iter(*inputs, **kwargs)
        elif mode == 'test':
            return self.test_iter(*inputs, **kwargs)
        elif mode == 'extract':
            return self.backbone(*inputs)
        else:
            raise Exception("No such mode: {}".format(mode))

        # initialize function
    def init_parameters(self, init_linear='kaiming', std=0.01, bias=0.):
        assert init_linear in ['normal', 'kaiming'], \
            "Undefined init_linear: {}".format(init_linear)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_init(m, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.layer.norm._BatchNormBase, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                if init_linear == 'normal':
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode='fan_in', nonlinearity='relu')


@paddle.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    """
    if paddle.distributed.get_world_size() < 2:
        return tensor

    tensors_gather = []
    paddle.distributed.all_gather(tensors_gather, tensor)

    output = paddle.concat(tensors_gather, axis=0)
    return output


def freeze_batchnorm_statictis(layer):
    def freeze_bn(layer):
        if isinstance(layer, (nn.layer.norm._BatchNormBase)):
            layer._use_global_stats = True
