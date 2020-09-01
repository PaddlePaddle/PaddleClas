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
    gpu_id = fluid.dygraph.parallel.Env().dev_id
    place = fluid.CUDAPlace(gpu_id)

    use_data_parallel = int(os.getenv("PADDLE_TRAINERS_NUM", 1)) != 1
    config["use_data_parallel"] = use_data_parallel

    with fluid.dygraph.guard(place):
        net = program.create_model(config.ARCHITECTURE, config.classes_num)
        if config["use_data_parallel"]:
            strategy = fluid.dygraph.parallel.prepare_context()
            net = fluid.dygraph.parallel.DataParallel(net, strategy)

        optimizer = program.create_optimizer(
            config, parameter_list=net.parameters())

        # load model from checkpoint or pretrained model
        init_model(config, net, optimizer)

        train_dataloader = program.create_dataloader()
        train_reader = Reader(config, 'train')()
        train_dataloader.set_sample_list_generator(train_reader, place)

        if config.validate:
            valid_dataloader = program.create_dataloader()
            valid_reader = Reader(config, 'valid')()
            valid_dataloader.set_sample_list_generator(valid_reader, place)

        best_top1_acc = 0.0  # best top1 acc record
        for epoch_id in range(config.epochs):
            net.train()
            # 1. train with train dataset
            program.run(train_dataloader, config, net, optimizer, epoch_id,
                        'train')

            if not config["use_data_parallel"] or fluid.dygraph.parallel.Env(
            ).local_rank == 0:
                # 2. validate with validate dataset
                if config.validate and epoch_id % config.valid_interval == 0:
                    net.eval()
                    top1_acc = program.run(valid_dataloader, config, net, None,
                                           epoch_id, 'valid')
                    if top1_acc > best_top1_acc:
                        best_top1_acc = top1_acc
                        message = "The best top1 acc {:.5f}, in epoch: {:d}".format(
                            best_top1_acc, epoch_id)
                        logger.info("{:s}".format(
                            logger.coloring(message, "RED")))
                        if epoch_id % config.save_interval == 0:

                            model_path = os.path.join(
                                config.model_save_dir,
                                config.ARCHITECTURE["name"])
                            save_model(net, optimizer, model_path,
                                       "best_model_in_epoch_" + str(epoch_id))

                # 3. save the persistable model
                if epoch_id % config.save_interval == 0:
                    model_path = os.path.join(config.model_save_dir,
                                              config.ARCHITECTURE["name"])
                    save_model(net, optimizer, model_path, epoch_id)


if __name__ == '__main__':
    args = parse_args()
    main(args)
