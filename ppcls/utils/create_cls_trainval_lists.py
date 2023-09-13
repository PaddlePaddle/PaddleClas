# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os
import random
import string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--save_img_list_path', type=str, default='train.txt')
    parser.add_argument('--save_label_map_path', type=str, default='label.txt')

    args = parser.parse_args()
    return args


def main(args):
    img_list = []
    label_list = []
    img_end = ['jpg', 'JPG', 'png', 'PNG', 'jpeg', 'JPEG', 'bmp']
    if args.dataset_path[-1] == "/":
        args.dataset_path = args.dataset_path[:-1]
    if not os.path.exists(args.dataset_path):
        raise Exception(f"The data path {args.dataset_path} not exists.")
    else:
        label_name_list = [
            label for label in os.listdir(args.dataset_path)
            if os.path.isdir(os.path.join(args.dataset_path, label))
        ]

    for index, label_name in enumerate(label_name_list):
        for root, dirs, files in os.walk(
                os.path.join(args.dataset_path, label_name)):
            for single_file in files:
                if single_file.split('.')[-1] in img_end:
                    img_path = os.path.relpath(
                        os.path.join(root, single_file),
                        os.path.dirname(args.dataset_path))
                    img_list.append(f'{img_path} {index}')
                else:
                    print(
                        f'WARNING: File {single_file} end with {single_file.split(".")[-1]} is not supported.'
                    )
        label_list.append(f'{index} {label_name}')

    if len(img_list) == 0:
        raise Exception(f"Not found any images file in {args.dataset_path}.")

    with open(
            os.path.join(
                os.path.dirname(args.dataset_path), args.save_img_list_path),
            'w') as f:
        f.write('\n'.join(img_list))
    print(
        f'Already save {args.save_img_list_path} in {os.path.join(os.path.dirname(args.dataset_path), args.save_img_list_path)}.'
    )

    with open(
            os.path.join(
                os.path.dirname(args.dataset_path), args.save_label_map_path),
            'w') as f:
        f.write('\n'.join(label_list))
    print(
        f'Already save {args.save_label_map_path} in {os.path.join(os.path.dirname(args.dataset_path), args.save_label_map_path)}.'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
