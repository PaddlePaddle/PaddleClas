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
from paddle.fluid.incubate.fleet.base import role_maker
from paddle.fluid.incubate.fleet.collective import fleet

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model, save_model
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
    role = role_maker.PaddleCloudRoleMaker(is_collective=True)
    fleet.init(role)

    config = get_config(args.config, overrides=args.override, show=True)
    # assign the place
    gpu_id = int(os.environ.get('FLAGS_selected_gpus', 0))
    place = fluid.CUDAPlace(gpu_id)

    # startup_prog is used to do some parameter init work,
    # and train prog is used to hold the network
    startup_prog = fluid.Program()
    train_prog = fluid.Program()

    train_dataloader, train_fetchs = program.build(
        config, train_prog, startup_prog, is_train=True)

    if config.validate:
        valid_prog = fluid.Program()
        valid_dataloader, valid_fetchs = program.build(
            config, valid_prog, startup_prog, is_train=False)
        # clone to prune some content which is irrelevant in valid_prog
        valid_prog = valid_prog.clone(for_test=True)

    # create the "Executor" with the statement of which place
    exe = fluid.Executor(place=place)
    # only run startup_prog once to init
    exe.run(startup_prog)

    # load model from checkpoint or pretrained model
    init_model(config, train_prog, exe)

    train_reader = Reader(config, 'train')()
    train_dataloader.set_sample_list_generator(train_reader, place)

    if config.validate:
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, place)
        compiled_valid_prog = program.compile(config, valid_prog)

    compiled_train_prog = fleet.main_program
    for epoch_id in range(config.epochs):
        # 1. train with train dataset
        program.run(train_dataloader, exe, compiled_train_prog, train_fetchs,
                    epoch_id, 'train')
        # 2. validate with validate dataset
        if config.validate and epoch_id % config.valid_interval == 0:
            program.run(valid_dataloader, exe, compiled_valid_prog,
                        valid_fetchs, epoch_id, 'valid')

        # 3. save the persistable model
        if epoch_id % config.save_interval == 0:
            model_path = os.path.join(config.model_save_dir,
                                      config.ARCHITECTURE["name"])
            save_model(train_prog, model_path, epoch_id)


if __name__ == '__main__':
    args = parse_args()
    main(args)
