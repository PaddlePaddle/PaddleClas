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
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

from sys import version_info

import paddle
import paddle.fluid as fluid

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model, save_model
from ppcls.utils import logger
import program


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
    config = get_config(args.config, overrides=args.override, show=True)
    # assign the place
    use_gpu = config.get("use_gpu", True)
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    best_top1_acc = 0.0  # best top1 acc record

    if not config.get('use_ema'):
        train_dataloader, train_fetchs = program.build(
            config,
            train_prog,
            startup_prog,
            is_train=True,
            is_distributed=False)
    else:
        train_dataloader, train_fetchs, ema = program.build(
            config,
            train_prog,
            startup_prog,
            is_train=True,
            is_distributed=False)

    if config.validate:
        valid_prog = fluid.Program()
        valid_dataloader, valid_fetchs = program.build(
            config,
            valid_prog,
            startup_prog,
            is_train=False,
            is_distributed=False)
        # clone to prune some content which is irrelevant in valid_prog
        valid_prog = valid_prog.clone(for_test=True)

    # create the "Executor" with the statement of which place
    exe = fluid.Executor(places[0])
    # Parameter initialization
    exe.run(startup_prog)

    # load model from 1. checkpoint to resume training, 2. pretrained model to finetune
    init_model(config, train_prog, exe)

    train_reader = Reader(config, 'train')()
    train_dataloader.set_sample_list_generator(train_reader, places)

    compiled_train_prog = program.compile(config, train_prog,
                                          train_fetchs['loss'][0].name)

    if config.validate:
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, places)
        compiled_valid_prog = program.compile(
            config, valid_prog, share_prog=compiled_train_prog)

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
        program.run(train_dataloader, exe, compiled_train_prog, train_fetchs,
                    epoch_id, 'train', config, vdl_writer)

        # 2. validate with validate dataset
        if config.validate and epoch_id % config.valid_interval == 0:
            if config.get('use_ema'):
                logger.info(logger.coloring("EMA validate start..."))
                with ema.apply(exe):
                    top1_acc = program.run(valid_dataloader, exe,
                                           compiled_valid_prog, valid_fetchs,
                                           epoch_id, 'valid', config)
                logger.info(logger.coloring("EMA validate over!"))

            top1_acc = program.run(valid_dataloader, exe, compiled_valid_prog,
                                   valid_fetchs, epoch_id, 'valid', config)
            if top1_acc > best_top1_acc:
                best_top1_acc = top1_acc
                message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                    best_top1_acc, epoch_id)
                logger.info("{:s}".format(logger.coloring(message, "RED")))
                if epoch_id % config.save_interval == 0:

                    model_path = os.path.join(config.model_save_dir,
                                              config.ARCHITECTURE["name"])
                    save_model(train_prog, model_path,
                               "best_model_in_epoch_" + str(epoch_id))

        # 3. save the persistable model
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir,
                                      config.ARCHITECTURE["name"])
            save_model(train_prog, model_path, epoch_id)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    main(args)
