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
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import argparse
import paddle
import paddle.fluid as fluid

import program

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model


def parse_args():
    parser = argparse.ArgumentParser("PaddleClas eval script")
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        default='./configs/eval.yaml',
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
    use_gpu = config.get("use_gpu", True)
    places = fluid.cuda_places() if use_gpu else fluid.cpu_places()

    startup_prog = fluid.Program()
    valid_prog = fluid.Program()
    valid_dataloader, valid_fetchs = program.build(
        config, valid_prog, startup_prog, is_train=False, is_distributed=False)
    valid_prog = valid_prog.clone(for_test=True)

    exe = fluid.Executor(places[0])
    exe.run(startup_prog)

    init_model(config, valid_prog, exe)

    valid_reader = Reader(config, 'valid')()
    valid_dataloader.set_sample_list_generator(valid_reader, places)

    compiled_valid_prog = program.compile(config, valid_prog)
    program.run(valid_dataloader, exe, compiled_valid_prog, valid_fetchs, -1,
                'eval', config)


if __name__ == '__main__':
    paddle.enable_static()
    args = parse_args()
    main(args)
