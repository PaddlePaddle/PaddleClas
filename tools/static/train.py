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

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from sys import version_info

import paddle
from paddle.distributed import fleet

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils import logger
from tools.static import program
from save_load import init_model, save_model


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas train script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='configs/ResNet/ResNet50.yaml',
        help='config file path')
    parser.add_argument(
        '--vdl_dir',
        type=str,
        default=None,
        help='VisualDL logging directory for image.')
    parser.add_argument(
        '-p',
        '--profiler_options',
        type=str,
        default=None,
        help='The option of profiler, which should be in format \"key1=value1;key2=value2;key3=value3\".'
    )
    parser.add_argument(
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args


def main(args):
    config = get_config(args.config, overrides=args.override, show=True)
    if config.get("is_distributed", True):
        fleet.init(is_collective=True)
    # assign the place
    use_gpu = config.get("use_gpu", True)
    # amp related config
    if 'AMP' in config:
        AMP_RELATED_FLAGS_SETTING = {
            'FLAGS_cudnn_exhaustive_search': 1,
            'FLAGS_conv_workspace_size_limit': 1500,
            'FLAGS_cudnn_batchnorm_spatial_persistent': 1,
            'FLAGS_max_inplace_grad_add': 8,
        }
        os.environ['FLAGS_cudnn_batchnorm_spatial_persistent'] = '1'
        paddle.fluid.set_flags(AMP_RELATED_FLAGS_SETTING)
    use_xpu = config.get("use_xpu", False)
    assert (
        use_gpu and use_xpu
    ) is not True, "gpu and xpu can not be true in the same time in static mode!"

    if use_gpu:
        place = paddle.set_device('gpu')
    elif use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = paddle.static.Program()
    train_prog = paddle.static.Program()

    best_top1_acc = 0.0  # best top1 acc record

    train_fetchs, lr_scheduler, train_feeds, optimizer = program.build(
        config,
        train_prog,
        startup_prog,
        is_train=True,
        is_distributed=config.get("is_distributed", True))

    if config.validate:
        valid_prog = paddle.static.Program()
        valid_fetchs, _, valid_feeds, _ = program.build(
            config,
            valid_prog,
            startup_prog,
            is_train=False,
            is_distributed=config.get("is_distributed", True))
        # clone to prune some content which is irrelevant in valid_prog
        valid_prog = valid_prog.clone(for_test=True)

    # create the "Executor" with the statement of which place
    exe = paddle.static.Executor(place)
    # Parameter initialization
    exe.run(startup_prog)
    # load pretrained models or checkpoints
    init_model(config, train_prog, exe)

    if 'AMP' in config and config.AMP.get("use_pure_fp16", False):
        optimizer.amp_init(
            place,
            scope=paddle.static.global_scope(),
            test_program=valid_prog if config.validate else None)

    if not config.get("is_distributed", True):
        compiled_train_prog = program.compile(
            config, train_prog, loss_name=train_fetchs["loss"][0].name)
    else:
        compiled_train_prog = train_prog

    if not config.get('use_dali', False):
        train_dataloader = Reader(config, 'train', places=place)()
        if config.validate and paddle.distributed.get_rank() == 0:
            valid_dataloader = Reader(config, 'valid', places=place)()
            compiled_valid_prog = program.compile(config, valid_prog)
    else:
        assert use_gpu is True, "DALI only support gpu, please set use_gpu to True!"
        import dali
        train_dataloader = dali.train(config)
        if config.validate and paddle.distributed.get_rank() == 0:
            valid_dataloader = dali.val(config)
            compiled_valid_prog = program.compile(config, valid_prog)

    vdl_writer = None
    if args.vdl_dir:
        if version_info.major == 2:
            logger.info(
                "visualdl is just supported for python3, so it is disabled in python2..."
            )
        else:
            from visualdl import LogWriter
            vdl_writer = LogWriter(args.vdl_dir)

    for epoch_id in range(config.epochs):
        # 1. train with train dataset
        program.run(train_dataloader, exe, compiled_train_prog, train_feeds,
                    train_fetchs, epoch_id, 'train', config, vdl_writer,
                    lr_scheduler, args.profiler_options)
        if paddle.distributed.get_rank() == 0:
            # 2. validate with validate dataset
            if config.validate and epoch_id % config.valid_interval == 0:
                top1_acc = program.run(valid_dataloader, exe,
                                       compiled_valid_prog, valid_feeds,
                                       valid_fetchs, epoch_id, 'valid', config)
                if top1_acc > best_top1_acc:
                    best_top1_acc = top1_acc
                    message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                        best_top1_acc, epoch_id)
                    logger.info("{:s}".format(logger.coloring(message, "RED")))
                    if epoch_id % config.save_interval == 0:

                        model_path = os.path.join(config.model_save_dir,
                                                  config.ARCHITECTURE["name"])
                        save_model(train_prog, model_path, "best_model")

            # 3. save the persistable model
            if epoch_id % config.save_interval == 0:
                model_path = os.path.join(config.model_save_dir,
                                          config.ARCHITECTURE["name"])
                save_model(train_prog, model_path, epoch_id)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    main(args)
