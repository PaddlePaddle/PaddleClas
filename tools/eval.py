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

import paddle
import paddle.nn.functional as F

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ppcls.utils import logger
from ppcls.utils.save_load import init_model
from ppcls.utils.config import get_config
from ppcls.utils import multi_hot_encode
from ppcls.utils import accuracy_score
from ppcls.utils import mean_average_precision
from ppcls.utils import precision_recall_fscore
from ppcls.data import Reader
import program

import numpy as np


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


def main(args, return_dict={}):
    config = get_config(args.config, overrides=args.override, show=True)
    config.mode = "valid"
    # assign place
    use_gpu = config.get("use_gpu", True)
    place = paddle.set_device('gpu' if use_gpu else 'cpu')
    multilabel = config.get("multilabel", False)

    trainer_num = paddle.distributed.get_world_size()
    use_data_parallel = trainer_num != 1
    config["use_data_parallel"] = use_data_parallel

    if config["use_data_parallel"]:
        paddle.distributed.init_parallel_env()

    net = program.create_model(config.ARCHITECTURE, config.classes_num)
    if config["use_data_parallel"]:
        net = paddle.DataParallel(net)

    init_model(config, net, optimizer=None)
    valid_dataloader = Reader(config, 'valid', places=place)()
    net.eval()
    with paddle.no_grad():
        if not multilabel:
            top1_acc = program.run(valid_dataloader, config, net, None, None, 0,
                                   'valid')
            return_dict["top1_acc"] = top1_acc

            return top1_acc
        else:
            all_outs = []
            targets = []
            for idx, batch in enumerate(valid_dataloader()):
                feeds = program.create_feeds(batch, False, config.classes_num, multilabel)
                out = net(feeds["image"])
                out = F.sigmoid(out)

                use_distillation = config.get("use_distillation", False)
                if use_distillation:
                    out = out[1]

                all_outs.extend(list(out.numpy()))
                targets.extend(list(feeds["label"].numpy()))
            all_outs = np.array(all_outs)
            targets = np.array(targets)

            mAP = mean_average_precision(all_outs, targets)

            return_dict["mean average precision"] = mAP

            return mAP


if __name__ == '__main__':
    args = parse_args()
    return_dict = {}
    main(args, return_dict)
    print(return_dict)
