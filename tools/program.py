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
import numpy as np

from collections import OrderedDict

import paddle.fluid as fluid

from ppcls.optimizer import LearningRateBuilder
from ppcls.optimizer import OptimizerBuilder
from ppcls.modeling import architectures
from ppcls.modeling.loss import CELoss
from ppcls.modeling.loss import MixCELoss
from ppcls.modeling.loss import JSDivLoss
from ppcls.modeling.loss import GoogLeNetLoss
from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger

from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.collective import DistributedStrategy

from ema import ExponentialMovingAverage


def create_feeds(image_shape, use_mix=None):
    """
    Create feeds as model input

    Args:
        image_shape(list[int]): model input shape, such as [3, 224, 224]
        use_mix(bool): whether to use mix(include mixup, cutmix, fmix)

    Returns:
        feeds(dict): dict of model input variables
    """
    feeds = OrderedDict()
    feeds['image'] = fluid.data(
        name="feed_image", shape=[None] + image_shape, dtype="float32")
    if use_mix:
        feeds['feed_y_a'] = fluid.data(
            name="feed_y_a", shape=[None, 1], dtype="int64")
        feeds['feed_y_b'] = fluid.data(
            name="feed_y_b", shape=[None, 1], dtype="int64")
        feeds['feed_lam'] = fluid.data(
            name="feed_lam", shape=[None, 1], dtype="float32")
    else:
        feeds['label'] = fluid.data(
            name="feed_label", shape=[None, 1], dtype="int64")

    return feeds


def create_dataloader(feeds):
    """
    Create a dataloader with model input variables

    Args:
        feeds(dict): dict of model input variables

    Returns:
        dataloader(fluid dataloader):
    """
    trainer_num = int(os.environ.get('PADDLE_TRAINERS_NUM', 1))
    capacity = 64 if trainer_num <= 1 else 8
    dataloader = fluid.io.DataLoader.from_generator(
        feed_list=feeds,
        capacity=capacity,
        use_double_buffer=True,
        iterable=True)

    return dataloader


def create_model(architecture, image, classes_num, is_train):
    """
    Create a model

    Args:
        architecture(dict): architecture information,
            name(such as ResNet50) is needed
        image(variable): model input variable
        classes_num(int): num of classes

    Returns:
        out(variable): model output variable
    """
    name = architecture["name"]
    params = architecture.get("params", {})
    if "is_test" in params:
        params['is_test'] = not is_train
    model = architectures.__dict__[name](**params)
    out = model.net(input=image, class_dim=classes_num)
    return out


def create_loss(out,
                feeds,
                architecture,
                classes_num=1000,
                epsilon=None,
                use_mix=False,
                use_distillation=False):
    """
    Create a loss for optimization, such as:
        1. CrossEnotry loss
        2. CrossEnotry loss with label smoothing
        3. CrossEnotry loss with mix(mixup, cutmix, fmix)
        4. CrossEnotry loss with label smoothing and (mixup, cutmix, fmix)
        5. GoogLeNet loss

    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables
        architecture(dict): architecture information,
            name(such as ResNet50) is needed
        classes_num(int): num of classes
        epsilon(float): parameter for label smoothing, 0.0 <= epsilon <= 1.0
        use_mix(bool): whether to use mix(include mixup, cutmix, fmix)

    Returns:
        loss(variable): loss variable
    """
    if architecture["name"] == "GoogLeNet":
        assert len(out) == 3, "GoogLeNet should have 3 outputs"
        loss = GoogLeNetLoss(class_dim=classes_num, epsilon=epsilon)
        target = feeds['label']
        return loss(out[0], out[1], out[2], target)

    if use_distillation:
        assert len(out) == 2, ("distillation output length must be 2, "
                               "but got {}".format(len(out)))
        loss = JSDivLoss(class_dim=classes_num, epsilon=epsilon)
        return loss(out[1], out[0])

    if use_mix:
        loss = MixCELoss(class_dim=classes_num, epsilon=epsilon)
        feed_y_a = feeds['feed_y_a']
        feed_y_b = feeds['feed_y_b']
        feed_lam = feeds['feed_lam']
        return loss(out, feed_y_a, feed_y_b, feed_lam)
    else:
        loss = CELoss(class_dim=classes_num, epsilon=epsilon)
        target = feeds['label']
        return loss(out, target)


def create_metric(out,
                  feeds,
                  architecture,
                  topk=5,
                  classes_num=1000,
                  use_distillation=False):
    """
    Create measures of model accuracy, such as top1 and top5

    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables(included label)
        topk(int): usually top5
        classes_num(int): num of classes

    Returns:
        fetchs(dict): dict of measures
    """
    if architecture["name"] == "GoogLeNet":
        assert len(out) == 3, "GoogLeNet should have 3 outputs"
        softmax_out = out[0]
    else:
        # just need student label to get metrics
        if use_distillation:
            out = out[1]
        softmax_out = fluid.layers.softmax(out, use_cudnn=False)

    fetchs = OrderedDict()
    # set top1 to fetchs
    top1 = fluid.layers.accuracy(softmax_out, label=feeds['label'], k=1)
    fetchs['top1'] = (top1, AverageMeter('top1', '.4f', need_avg=True))
    # set topk to fetchs
    k = min(topk, classes_num)
    topk = fluid.layers.accuracy(softmax_out, label=feeds['label'], k=k)
    topk_name = 'top{}'.format(k)
    fetchs[topk_name] = (topk, AverageMeter(topk_name, '.4f', need_avg=True))

    return fetchs


def create_fetchs(out,
                  feeds,
                  architecture,
                  topk=5,
                  classes_num=1000,
                  epsilon=None,
                  use_mix=False,
                  use_distillation=False):
    """
    Create fetchs as model outputs(included loss and measures),
    will call create_loss and create_metric(if use_mix).

    Args:
        out(variable): model output variable
        feeds(dict): dict of model input variables.
            If use mix_up, it will not include label.
        architecture(dict): architecture information,
            name(such as ResNet50) is needed
        topk(int): usually top5
        classes_num(int): num of classes
        epsilon(float): parameter for label smoothing, 0.0 <= epsilon <= 1.0
        use_mix(bool): whether to use mix(include mixup, cutmix, fmix)

    Returns:
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    fetchs = OrderedDict()
    loss = create_loss(out, feeds, architecture, classes_num, epsilon, use_mix,
                       use_distillation)
    fetchs['loss'] = (loss, AverageMeter('loss', '7.4f', need_avg=True))
    if not use_mix:
        metric = create_metric(out, feeds, architecture, topk, classes_num,
                               use_distillation)
        fetchs.update(metric)

    return fetchs


def create_optimizer(config):
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
    return opt(lr)


def dist_optimizer(config, optimizer):
    """
    Create a distributed optimizer based on a normal optimizer

    Args:
        config(dict):
        optimizer(): a normal optimizer

    Returns:
        optimizer: a distributed optimizer
    """
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_threads = 3
    exec_strategy.num_iteration_per_drop_scope = 10

    dist_strategy = DistributedStrategy()
    dist_strategy.nccl_comm_num = 1
    dist_strategy.fuse_all_reduce_ops = True
    dist_strategy.exec_strategy = exec_strategy
    optimizer = fleet.distributed_optimizer(optimizer, strategy=dist_strategy)

    return optimizer


def mixed_precision_optimizer(config, optimizer):
    use_fp16 = config.get('use_fp16', False)
    amp_scale_loss = config.get('amp_scale_loss', 1.0)
    use_dynamic_loss_scaling = config.get('use_dynamic_loss_scaling', False)
    if use_fp16:
        optimizer = fluid.contrib.mixed_precision.decorate(
            optimizer,
            init_loss_scaling=amp_scale_loss,
            use_dynamic_loss_scaling=use_dynamic_loss_scaling)

    return optimizer


def build(config, main_prog, startup_prog, is_train=True, is_distributed=True):
    """
    Build a program using a model and an optimizer
        1. create feeds
        2. create a dataloader
        3. create a model
        4. create fetchs
        5. create an optimizer

    Args:
        config(dict): config
        main_prog(): main program
        startup_prog(): startup program
        is_train(bool): train or valid
        is_distributed(bool): whether to use distributed training method

    Returns:
        dataloader(): a bridge between the model and the data
        fetchs(dict): dict of model outputs(included loss and measures)
    """
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            use_mix = config.get('use_mix') and is_train
            use_distillation = config.get('use_distillation')
            feeds = create_feeds(config.image_shape, use_mix=use_mix)
            dataloader = create_dataloader(feeds.values())
            out = create_model(config.ARCHITECTURE, feeds['image'],
                               config.classes_num, is_train)
            fetchs = create_fetchs(
                out,
                feeds,
                config.ARCHITECTURE,
                config.topk,
                config.classes_num,
                epsilon=config.get('ls_epsilon'),
                use_mix=use_mix,
                use_distillation=use_distillation)
            if is_train:
                optimizer = create_optimizer(config)
                lr = optimizer._global_learning_rate()
                fetchs['lr'] = (lr, AverageMeter('lr', 'f', need_avg=False))

                optimizer = mixed_precision_optimizer(config, optimizer)
                if is_distributed:
                    optimizer = dist_optimizer(config, optimizer)
                optimizer.minimize(fetchs['loss'][0])
                if config.get('use_ema'):

                    global_steps = fluid.layers.learning_rate_scheduler._decay_step_counter(
                    )
                    ema = ExponentialMovingAverage(
                        config.get('ema_decay'), thres_steps=global_steps)
                    ema.update()
                    return dataloader, fetchs, ema

    return dataloader, fetchs


def compile(config, program, loss_name=None, share_prog=None):
    """
    Compile the program

    Args:
        config(dict): config
        program(): the program which is wrapped by
        loss_name(str): loss name
        share_prog(): the shared program, used for evaluation during training

    Returns:
        compiled_program(): a compiled program
    """
    build_strategy = fluid.compiler.BuildStrategy()
    exec_strategy = fluid.ExecutionStrategy()

    exec_strategy.num_threads = 1
    exec_strategy.num_iteration_per_drop_scope = 10

    compiled_program = fluid.CompiledProgram(program).with_data_parallel(
        share_vars_from=share_prog,
        loss_name=loss_name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)

    return compiled_program


total_step = 0


def run(dataloader,
        exe,
        program,
        fetchs,
        epoch=0,
        mode='train',
        config=None,
        vdl_writer=None):
    """
    Feed data to the model and fetch the measures and loss

    Args:
        dataloader(fluid dataloader):
        exe():
        program():
        fetchs(dict): dict of measures and the loss
        epoch(int): epoch of training or validation
        model(str): log only

    Returns:
    """
    fetch_list = [f[0] for f in fetchs.values()]
    metric_list = [f[1] for f in fetchs.values()]
    for m in metric_list:
        m.reset()
    batch_time = AverageMeter('elapse', '.3f')
    tic = time.time()
    for idx, batch in enumerate(dataloader()):
        metrics = exe.run(program=program, feed=batch, fetch_list=fetch_list)
        batch_time.update(time.time() - tic)
        tic = time.time()
        for i, m in enumerate(metrics):
            metric_list[i].update(np.mean(m), len(batch[0]))
        fetchs_str = ''.join([str(m.value) + ' '
                              for m in metric_list] + [batch_time.value]) + 's'
        if vdl_writer:
            global total_step
            logger.scaler('loss', metrics[0][0], total_step, vdl_writer)
            total_step += 1
        if mode == 'eval':
            if idx % config.get('print_interval', 10) == 0:
                logger.info("{:s} step:{:<4d} {:s}".format(mode, idx,
                                                           fetchs_str))
        else:
            epoch_str = "epoch:{:<3d}".format(epoch)
            step_str = "{:s} step:{:<4d}".format(mode, idx)

            # Keep the first 10 batches statistics, They are important for develop
            if epoch == 0 and idx < 10:
                logger.info("{:s} {:s} {:s}".format(
                    logger.coloring(epoch_str, "HEADER")
                    if idx == 0 else epoch_str,
                    logger.coloring(step_str, "PURPLE"),
                    logger.coloring(fetchs_str, 'OKGREEN')))

            else:
                if idx % config.get('print_interval', 10) == 0:
                    logger.info("{:s} {:s} {:s}".format(
                        logger.coloring(epoch_str, "HEADER")
                        if idx == 0 else epoch_str,
                        logger.coloring(step_str, "PURPLE"),
                        logger.coloring(fetchs_str, 'OKGREEN')))

    end_str = ''.join([str(m.mean) + ' '
                       for m in metric_list] + [batch_time.total]) + 's'
    if mode == 'eval':
        logger.info("END {:s} {:s}s".format(mode, end_str))
    else:
        end_epoch_str = "END epoch:{:<3d}".format(epoch)

        logger.info("{:s} {:s} {:s}".format(
            logger.coloring(end_epoch_str, "RED"),
            logger.coloring(mode, "PURPLE"),
            logger.coloring(end_str, "OKGREEN")))

    # return top1_acc in order to save the best model
    if mode == 'valid':
        return fetchs["top1"][1].avg
