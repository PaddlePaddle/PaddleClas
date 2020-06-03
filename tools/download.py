# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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

from ppcls import model_zoo


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--architecture', type=str, default='ResNet50')
    parser.add_argument('-p', '--path', type=str, default='./pretrained/')
    parser.add_argument('--postfix', type=str, default="tar")
    parser.add_argument('-d', '--decompress', type=str2bool, default=True)
    parser.add_argument('-l', '--list', type=str2bool, default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.list:
        model_zoo.list_models()
    else:
        model_zoo.get(args.architecture, args.path, args.decompress,
                      args.postfix)


if __name__ == '__main__':
    main()
