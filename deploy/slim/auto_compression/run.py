# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import os
import sys
import argparse
import functools
from functools import partial
import math
from tqdm import tqdm

import numpy as np
import paddle
import paddleslim
from paddle.jit import to_static
from paddleslim.analysis import dygraph_flops as flops

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))
from paddleslim.auto_compression import AutoCompression

from ppcls.data import build_dataloader
from ppcls.utils import config as conf
from ppcls.utils.logger import init_logger


def reader_wrapper(reader, input_name):
    def gen():
        for i, (imgs, label) in enumerate(reader()):
            yield {input_name: imgs}

    return gen


def eval_function(exe, compiled_test_program, test_feed_names,
                  test_fetch_list):
    results = []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, (image, label) in enumerate(val_loader):

            # top1_acc, top5_acc
            if len(test_feed_names) == 1:
                image = np.array(image)
                label = np.array(label).astype('int64')
                pred = exe.run(compiled_test_program,
                               feed={test_feed_names[0]: image},
                               fetch_list=test_fetch_list)
                pred = np.array(pred[0])
                label = np.array(label).reshape((-1, 1))
                sort_array = pred.argsort(axis=1)
                top_1_pred = sort_array[:, -1:][:, ::-1]
                top_1 = np.mean(label == top_1_pred)
                top_5_pred = sort_array[:, -5:][:, ::-1]
                acc_num = 0
                for i in range(len(label)):
                    if label[i][0] in top_5_pred[i]:
                        acc_num += 1
                top_5 = float(acc_num) / len(label)
                results.append([top_1, top_5])
            else:
                # eval "eval model", which inputs are image and label, output is top1 and top5 accuracy
                image = np.array(image)
                label = np.array(label).astype('int64')
                result = exe.run(compiled_test_program,
                                 feed={
                                     test_feed_names[0]: image,
                                     test_feed_names[1]: label
                                 },
                                 fetch_list=test_fetch_list)
                result = [np.mean(r) for r in result]
                results.append(result)
            t.update()
    result = np.mean(np.array(results), axis=0)
    return result[0]


def main():
    args = conf.parse_args()
    global config
    config = conf.get_config(args.config, overrides=args.override, show=False)

    assert os.path.exists(
        os.path.join(config["Global"]["model_dir"], 'inference.pdmodel')
    ) and os.path.exists(
        os.path.join(config["Global"]["model_dir"], 'inference.pdiparams'))
    if "Query" in config["DataLoader"]["Eval"]:
        config["DataLoader"]["Eval"] = config["DataLoader"]["Eval"]["Query"]

    init_logger()
    train_dataloader = build_dataloader(config["DataLoader"], "Train",
                                        config["Global"]['device'], False)
    if isinstance(config['TrainConfig']['learning_rate'], dict) and config[
            'TrainConfig']['learning_rate']['type'] == 'CosineAnnealingDecay':

        gpu_num = paddle.distributed.get_world_size()
        step = len(train_dataloader)
        config['TrainConfig']['learning_rate']['T_max'] = step
        print('total training steps:', step)

    global val_loader
    val_loader = build_dataloader(config["DataLoader"], "Eval",
                                  config["Global"]['device'], False)

    if config["Global"]['device'] == 'gpu':
        rank_id = paddle.distributed.get_rank()
        place = paddle.CUDAPlace(rank_id)
        paddle.set_device('gpu')
    else:
        place = paddle.CPUPlace()
        paddle.set_device('cpu')

    ac = AutoCompression(
        model_dir=config["Global"]["model_dir"],
        model_filename=config["Global"]["model_filename"],
        params_filename=config["Global"]["params_filename"],
        save_dir=config["Global"]['output_dir'],
        config=config,
        train_dataloader=reader_wrapper(
            train_dataloader, input_name=config['Global']['input_name']),
        eval_callback=eval_function if rank_id == 0 else None,
        eval_dataloader=reader_wrapper(
            val_loader, input_name=config['Global']['input_name']))

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    main()
