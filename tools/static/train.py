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
from paddle.distributed import ParallelEnv
from paddle.distributed import fleet

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils import logger
from tools.static import program
from program import save_model


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
        '-o',
        '--override',
        action='append',
        default=[],
        help='config options to be overridden')
    args = parser.parse_args()
    return args


def main(args):
    fleet.init(is_collective=True)

    config = get_config(args.config, overrides=args.override, show=True)
    # assign the place
    use_gpu = config.get("use_gpu", True)
    assert use_gpu is True, "gpu must be true in static mode!"
    place = 'gpu:{}'.format(ParallelEnv().dev_id)
    place = paddle.set_device(place)

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = paddle.static.Program()
    train_prog = paddle.static.Program()

    best_top1_acc = 0.0  # best top1 acc record

    train_fetchs, lr_scheduler, train_feeds = program.build(
        config, train_prog, startup_prog, is_train=True)

    if config.validate:
        valid_prog = paddle.static.Program()
        valid_fetchs, _, valid_feeds = program.build(
            config, valid_prog, startup_prog, is_train=False)
        # clone to prune some content which is irrelevant in valid_prog
        valid_prog = valid_prog.clone(for_test=True)

    # create the "Executor" with the statement of which place
    exe = paddle.static.Executor(place)
    # Parameter initialization
    exe.run(startup_prog)

    # load model from 1. checkpoint to resume training, 2. pretrained model to finetune
    train_dataloader = Reader(config, 'train', places=place)()

    if config.validate and ParallelEnv().local_rank == 0:
        valid_dataloader = Reader(config, 'valid', places=place)()
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
        program.run(train_dataloader, exe, train_prog, train_feeds, train_fetchs, epoch_id,
                    'train', config, vdl_writer, lr_scheduler)
        if int(os.getenv("PADDLE_TRAINER_ID", 0)) == 0:
            # 2. validate with validate dataset
            if config.validate and epoch_id % config.valid_interval == 0:
                top1_acc = program.run(valid_dataloader, exe,
                                       compiled_valid_prog, valid_feeds, valid_fetchs,
                                       epoch_id, 'valid', config)
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
