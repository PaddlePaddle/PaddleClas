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

import os
import sys
import math
import argparse
import numpy as np
from tqdm import tqdm

import paddle
from paddleslim.common import load_config as load_slim_config
from paddleslim.auto_compression import AutoCompression

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))
from ppcls.data import build_dataloader
from ppcls.utils import config
from ppcls.utils.logger import init_logger


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--compression_config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--reader_config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--total_images',
        type=int,
        default=1281167,
        help="the number of total training images.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")
    return parser


# yapf: enable
def reader_wrapper(reader, input_name):
    if isinstance(input_name, list) and len(input_name) == 1:
        input_name = input_name[0]

    def gen():
        for i, (imgs, label) in enumerate(reader()):
            yield {input_name: imgs}

    return gen


def eval_function(exe, compiled_test_program, test_feed_names,
                  test_fetch_list):

    results = []
    with tqdm(
            total=len(eval_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for batch_id, (image, label) in enumerate(eval_loader):
            # top1_acc, top5_acc
            if len(test_feed_names) == 1:
                image = np.array(image)
                label = np.array(label).astype('int64')
                if len(label.shape) == 1:
                    label = label.reshape([label.shape[0], -1])
                pred = exe.run(compiled_test_program,
                               feed={test_feed_names[0]: image},
                               fetch_list=test_fetch_list)
                pred = np.array(pred[0])
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
    rank_id = paddle.distributed.get_rank()
    if args.devices == 'gpu':
        paddle.CUDAPlace(rank_id)
        device = paddle.set_device('gpu')
    else:
        paddle.CPUPlace()
        device = paddle.set_device('cpu')
    global global_config
    all_config = load_slim_config(args.compression_config_path)

    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]

    gpu_num = paddle.distributed.get_world_size()
    if isinstance(all_config['TrainConfig']['learning_rate'],
                  dict) and all_config['TrainConfig']['learning_rate'][
                      'type'] == 'CosineAnnealingDecay':
        step = int(
            math.ceil(
                float(args.total_images) / (global_config['batch_size'] *
                                            gpu_num)))
        all_config['TrainConfig']['learning_rate']['T_max'] = step
        print('total training steps:', step)

    init_logger()
    data_config = config.get_config(args.reader_config_path, show=False)
    train_loader = build_dataloader(data_config["DataLoader"], "Train", device,
                                    False)
    train_dataloader = reader_wrapper(train_loader,
                                      global_config['input_name'])

    global eval_loader
    eval_loader = build_dataloader(data_config["DataLoader"], "Eval", device,
                                   False)
    eval_dataloader = reader_wrapper(eval_loader, global_config['input_name'])

    ac = AutoCompression(
        model_dir=global_config['model_dir'],
        model_filename=global_config['model_filename'],
        params_filename=global_config['params_filename'],
        save_dir=args.save_dir,
        config=all_config,
        train_dataloader=train_dataloader,
        eval_callback=eval_function if rank_id == 0 else None,
        eval_dataloader=eval_dataloader)

    ac.compress()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    args = parser.parse_args()
    main()
