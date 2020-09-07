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
import program
from ppcls.utils import logger
from ppcls.utils.save_load import init_model
from ppcls.utils.config import get_config
from ppcls.data import Reader
import paddle.fluid as fluid
import paddle
import argparse
from __future__ import division
from __future__ import print_function

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))


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
    # assign place
    use_gpu = config.get("use_gpu", True)
    if use_gpu:
        gpu_id = fluid.dygraph.ParallelEnv().dev_id
        place = fluid.CUDAPlace(gpu_id)
    else:
        place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        strategy = fluid.dygraph.parallel.prepare_context()
        net = program.create_model(config.ARCHITECTURE, config.classes_num)
        net = fluid.dygraph.parallel.DataParallel(net, strategy)
        init_model(config, net, optimizer=None)
        valid_dataloader = program.create_dataloader()
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, place)
        net.eval()
        top1_acc = program.run(valid_dataloader, config, net, None, 0, 'valid')


if __name__ == '__main__':
    args = parse_args()
    main(args)
