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
import argparse

import paddle.fluid as fluid

import program

from ppcls.data import Reader
from ppcls.utils.config import get_config
from ppcls.utils.save_load import init_model
from ppcls.utils import logger

from paddle.fluid.incubate.fleet.collective import fleet
from paddle.fluid.incubate.fleet.base import role_maker


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
    # assign the place
    gpu_id = fluid.dygraph.parallel.Env().dev_id
    place = fluid.CUDAPlace(gpu_id)

    with fluid.dygraph.guard(place):
        pre_weights_dict = fluid.dygraph.load_dygraph(config.pretrained_model +
                                                      "/ppcls")[0]
        strategy = fluid.dygraph.parallel.prepare_context()
        net = program.create_model(config.ARCHITECTURE, config.classes_num)
        net = fluid.dygraph.parallel.DataParallel(net, strategy)
        net.set_dict(pre_weights_dict)

        valid_dataloader = program.create_dataloader()
        valid_reader = Reader(config, 'valid')()
        valid_dataloader.set_sample_list_generator(valid_reader, place)

        net.eval()
        top1_acc = program.run(valid_dataloader, config, net, None, 0, 'valid')


if __name__ == '__main__':
    args = parse_args()
    main(args)
