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

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

import paddle

from multiprocessing import Process, Manager
import threading
import tools.eval as eval
from ppcls.utils.model_zoo import _download, _decompress
from ppcls.utils import logger


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b",
        "--benchmark_file_list",
        type=str,
        default="./tools/benchmark/benchmark_list.txt")
    parser.add_argument(
        "-p", "--pretrained_dir", type=str, default="./pretrained/")

    return parser.parse_args()


def parse_model_infos(benchmark_file_list):
    model_infos = []
    with open(benchmark_file_list, "r") as fin:
        lines = fin.readlines()
        for idx, line in enumerate(lines):
            strs = line.strip("\n").strip("\r").split(" ")
            if len(strs) != 4:
                logger.info(
                    "line {0}(info: {1}) format wrong, it should be splited into 4 parts, but got {2}".
                    format(idx, line, len(strs)))
            model_infos.append({
                "top1_acc": float(strs[0]),
                "model_name": strs[1],
                "config_path": strs[2],
                "pretrain_path": strs[3],
            })
    return model_infos


def main(args):
    benchmark_file_list = args.benchmark_file_list
    model_infos = parse_model_infos(benchmark_file_list)
    right_models = []
    wrong_models = []

    for model_info in model_infos:
        try:
            pretrained_url = model_info["pretrain_path"]
            fname = _download(pretrained_url, args.pretrained_dir)
            pretrained_path = os.path.splitext(fname)[0]
            if pretrained_url.endswith("tar"):
                path = _decompress(fname)
                pretrained_path = os.path.join(
                    os.path.dirname(pretrained_path), path)

            args.config = model_info["config_path"]
            args.override = [
                "pretrained_model={}".format(pretrained_path),
                "VALID.batch_size=256",
                "VALID.num_workers=16",
                "load_static_weights=True",
                "print_interval=100",
            ]

            manager = Manager()
            return_dict = manager.dict()

            # A hack method to avoid name conflict.
            # Multi-process maybe a better method here.
            # More details can be seen in branch 2.0-beta.
            # TODO: fluid needs to be removed in the future.
            with paddle.utils.unique_name.guard():
                eval.main(args, return_dict)

            top1_acc = return_dict.get("top1_acc", 0.0)
        except Exception as e:
            logger.error(e)
            top1_acc = 0.0
        diff = abs(top1_acc - model_info["top1_acc"])
        if diff > 0.001:
            err_info = "[{}]Top-1 acc diff should be <= 0.001 but got diff {}, gt acc: {}, eval acc: {}".format(
                model_info["model_name"], diff, model_info["top1_acc"],
                top1_acc)
            logger.warning(err_info)
            wrong_models.append(model_info["model_name"])
        else:
            right_models.append(model_info["model_name"])

    logger.info("[number of right models: {}, they are: {}".format(
        len(right_models), right_models))
    logger.info("[number of wrong models: {}, they are: {}".format(
        len(wrong_models), wrong_models))


if __name__ == '__main__':
    args = parse_args()
    main(args)
