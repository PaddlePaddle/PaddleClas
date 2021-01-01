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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from collections import OrderedDict

import paddle
from paddle import to_tensor
import paddle.nn as nn
import paddle.nn.functional as F

from ppcls.optimizer import LearningRateBuilder
from ppcls.optimizer import OptimizerBuilder
from ppcls.modeling import architectures
from ppcls.modeling.loss import LossBuilder
from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def create_model(architecture, classes_num):
    """
    Create a model

    Args:
        architecture(dict): architecture information,
            name(such as ResNet50) is needed
        classes_num(int): num of classes

    Returns:
        out(nn.Layer): model output tensor
    """
    name = architecture["name"]
    params = architecture.get("params", {})
    return architectures.__dict__[name](class_dim=classes_num, **params)


def create_loss(config):
    """
    Build loss function

    Args:
        config(dict): config
    
    Returns:
        loss_func(nn.Layer): loss function
    """
    classes_num = config["classes_num"]
    params = config["LOSS"]["params"]
    if not isinstance(params, (dict, )):
        params = {}
    params["classes_num"] = config["classes_num"]

    loss_func = LossBuilder(
        function=config["LOSS"]["function"], params=params)()
    return loss_func


def create_metric(out,
                  label,
                  architecture,
                  topk=5,
                  classes_num=1000,
                  use_distillation=False,
                  mode="train"):
    """
    Create measures of model accuracy, such as top1 and top5

    Args:
        out(tensor): model output tensor
        feeds(dict): dict of model input tensors(included label)
        topk(int): usually top5
        classes_num(int): num of classes
        use_distillation(bool): whether to use distillation training
        mode(str): mode, train/valid

    Returns:
        fetchs(dict): dict of measures
    """
    if architecture["name"] == "GoogLeNet":
        assert len(out) == 3, "GoogLeNet should have 3 outputs"
        out = out[0]
    else:
        # just need student label to get metrics
        if use_distillation:
            out = out[1]
    softmax_out = F.softmax(out)

    fetchs = OrderedDict()
    # set top1 to fetchs
    top1 = paddle.metric.accuracy(softmax_out, label=label, k=1)
    # set topk to fetchs
    k = min(topk, classes_num)
    topk = paddle.metric.accuracy(softmax_out, label=label, k=k)

    # multi cards' eval
    if mode != "train" and paddle.distributed.get_world_size() > 1:
        top1 = paddle.distributed.all_reduce(
            top1, op=paddle.distributed.ReduceOp.
            SUM) / paddle.distributed.get_world_size()
        topk = paddle.distributed.all_reduce(
            topk, op=paddle.distributed.ReduceOp.
            SUM) / paddle.distributed.get_world_size()

    fetchs['top1'] = top1
    topk_name = 'top{}'.format(k)
    fetchs[topk_name] = topk

    return fetchs


def create_fetchs(feeds, net, config, loss_func, mode="train"):
    """
    Create fetchs as model outputs(included loss and measures),
    will call create_metric(if use_mix).

    Args:
        feeds(dict): dict of model input tensors.
            If use mix_up, it will not include label.
        net(dict): network
        config(dict): root config for training/eval process
        loss_func(func): loss function of the network
        mode(string): run mode, train/valid
    Returns:
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    architecture = config.ARCHITECTURE
    topk = config.topk
    classes_num = config.classes_num
    use_mix = config.get('use_mix') and mode == 'train'
    use_distillation = config.get('use_distillation')

    out = net(feeds["image"])

    fetchs = OrderedDict()

    if use_distillation:
        assert len(out) == 2, ("distillation output length must be 2, "
                               "but got {}".format(len(out)))
        x = out[1]
        target = out[0]
    else:
        x = out
        target = feeds
    fetchs['loss'] = loss_func(x, target, mode=mode)

    if not use_mix:
        metric = create_metric(
            out,
            feeds["label"],
            architecture,
            topk,
            classes_num,
            use_distillation,
            mode=mode)
        fetchs.update(metric)

    return fetchs


def create_optimizer(config, parameter_list=None):
    """
    Create an optimizer using config, usually including
    learning rate and regularization.

    Args:
        config(dict):  such as
        {
            'LEARNING_RATE':
                {'function': 'Cosine',
                 'params': {'lr': 0.1}
                },
            'OPTIMIZER':
                {'function': 'Momentum',
                 'params':{'momentum': 0.9},
                 'regularizer':
                    {'function': 'L2', 'factor': 0.0001}
                }
        }

        parameter_list(list): parameter list

    Returns:
        an optimizer instance
    """
    # create learning_rate instance
    lr_config = config['LEARNING_RATE']
    lr_config['params'].update({
        'epochs': config['epochs'],
        'step_each_epoch':
        config['total_images'] // config['TRAIN']['batch_size'],
    })
    lr = LearningRateBuilder(**lr_config)()

    # create optimizer instance
    opt_config = config['OPTIMIZER']
    opt = OptimizerBuilder(**opt_config)
    return opt(lr, parameter_list), lr


def create_feeds(batch, use_mix):
    """
    Create input batch

    Args:
        batch(list):  batch data, including image and label
        use_mix(bool): whether to use mix for training
    Returns:
        feeds(dict): dict of model inputs
            1. for use_mix=False, dict keys are image, label
            2. for use_mix=True,  dict keys are image, y_a, y_b, lam
    """
    image = batch[0]
    if use_mix:
        y_a = to_tensor(batch[1].numpy().astype("int64").reshape(-1, 1))
        y_b = to_tensor(batch[2].numpy().astype("int64").reshape(-1, 1))
        lam = to_tensor(batch[3].numpy().astype("float32").reshape(-1, 1))
        feeds = {"image": image, "y_a": y_a, "y_b": y_b, "lam": lam}
    else:
        label = to_tensor(batch[1].numpy().astype('int64').reshape(-1, 1))
        feeds = {"image": image, "label": label}
    return feeds


def run(dataloader,
        config,
        net,
        optimizer=None,
        lr_scheduler=None,
        epoch=0,
        mode='train'):
    """
    Feed data to the model and fetch the measures and loss

    Args:
        dataloader(paddle dataloader):
        config(dict): config
        net(nn.Layer): network
        optimizer(): optimizer instance
        lr_scheduler(): learning rate schedulers instance
        epoch(int): epoch num
        mode(string): run mode, train/valid
        fetchs(dict): dict of measures and the loss
        epoch(int): epoch of training or validation
        model(str): log only

    Returns:
        None
    """
    print_interval = config.get("print_interval", 10)
    use_mix = config.get("use_mix", False) and mode == "train"

    metric_list = [
        ("loss", AverageMeter(
            'loss', '7.5f', postfix=",")),
        ("lr", AverageMeter(
            'lr', 'f', postfix=",", need_avg=False)),
        ("batch_time", AverageMeter(
            'batch_cost', '.5f', postfix=" s,")),
        ("reader_time", AverageMeter(
            'reader_cost', '.5f', postfix=" s,")),
    ]
    if not use_mix:
        topk_name = 'top{}'.format(config.topk)
        metric_list.insert(
            1, (topk_name, AverageMeter(
                topk_name, '.5f', postfix=",")))
        metric_list.insert(
            1, ("top1", AverageMeter(
                "top1", '.5f', postfix=",")))

    metric_list = OrderedDict(metric_list)

    loss_func = create_loss(config)

    tic = time.time()
    for idx, batch in enumerate(dataloader()):
        # avoid statistics from warmup time
        if idx == 10:
            metric_list["batch_time"].reset()
            metric_list["reader_time"].reset()

        metric_list['reader_time'].update(time.time() - tic)
        batch_size = len(batch[0])
        feeds = create_feeds(batch, use_mix)
        fetchs = create_fetchs(feeds, net, config, loss_func, mode)
        if mode == 'train':
            avg_loss = fetchs['loss']
            avg_loss.backward()

            optimizer.step()
            optimizer.clear_grad()
            metric_list['lr'].update(
                optimizer._global_learning_rate().numpy()[0], batch_size)

            if lr_scheduler is not None:
                if lr_scheduler.update_specified:
                    curr_global_counter = lr_scheduler.step_each_epoch * epoch + idx
                    update = max(
                        0, curr_global_counter - lr_scheduler.update_start_step
                    ) % lr_scheduler.update_step_interval == 0
                    if update:
                        lr_scheduler.step()
                else:
                    lr_scheduler.step()

        for name, fetch in fetchs.items():
            metric_list[name].update(fetch.numpy()[0], batch_size)
        metric_list["batch_time"].update(time.time() - tic)
        tic = time.time()

        fetchs_str = ' '.join([
            str(metric_list[key].mean)
            if "time" in key else str(metric_list[key].value)
            for key in metric_list
        ])

        if idx % print_interval == 0:
            ips_info = "ips: {:.5f} images/sec.".format(
                batch_size / metric_list["batch_time"].avg)
            if mode == 'eval':
                logger.info("{:s} step:{:<4d}, {:s} {:s}".format(
                    mode, idx, fetchs_str, ips_info))
            else:
                epoch_str = "epoch:{:<3d}".format(epoch)
                step_str = "{:s} step:{:<4d}".format(mode, idx)
                logger.info("{:s}, {:s}, {:s} {:s}".format(
                    epoch_str, step_str, fetchs_str, ips_info))

    end_str = " ".join([str(m.mean) for m in metric_list.values()] +
                       [metric_list['batch_time'].total])
    ips_info = "ips: {:.5f} images/sec.".format(
        batch_size * metric_list["batch_time"].count /
        metric_list["batch_time"].sum)

    if mode == "valid":
        logger.info("END {:s} {:s} {:s}".format(mode, end_str, ips_info))
    else:
        end_epoch_str = "END epoch:{:<3d}".format(epoch)

        logger.info("{:s} {:s} {:s} {:s}".format(end_epoch_str, mode, end_str,
                                                 ips_info))

    # return top1_acc in order to save the best model
    if mode == "valid":
        return metric_list['top1'].avg
