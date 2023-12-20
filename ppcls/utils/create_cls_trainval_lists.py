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
from random import shuffle
import string


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./data')
    parser.add_argument('--train_list_rate', type=int, default=80)
    parser.add_argument('--val_list_rate', type=int, default=20)
    parser.add_argument('--test_list_rate', type=int, default=0)
    args = parser.parse_args()
    return args


def parse_class_id_map(class_id_map_file):
    class_id_map = {}
    with open(class_id_map_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            partition = line.split("\n")[0].partition(" ")
            class_id_map[str(partition[-1])] = int(partition[0])
    return class_id_map


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

    sum_rate = args.train_list_rate + args.val_list_rate + args.test_list_rate
    if sum_rate != 100:
        raise Exception("训练集、验证集、测试集比例之和需要等于100，请修改后重试")
    tags = ["train", "val", "test"]

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
                        f'WARNING: File {os.path.join(root, single_file)} end with {single_file.split(".")[-1]} is not supported.'
                    )
        label_list.append(f'{index} {label_name}')

    shuffle(img_list)
    if len(img_list) == 0:
        raise Exception(f"Not found any images file in {args.dataset_path}.")

    start = 0
    image_num = len(img_list)
    rate_list = [args.train_list_rate, args.val_list_rate, args.test_list_rate]

    for i, tag in enumerate(tags):
        rate = rate_list[i]
        if rate == 0:
            continue
        if rate > 100 or rate < 0:
            return f"{tag} 数据集的比例应该在0~100之间."

        end = start + round(image_num * rate / 100)
        if sum(rate_list[i + 1:]) == 0:
            end = image_num

        txt_file = os.path.abspath(
            os.path.join(os.path.dirname(args.dataset_path), tag + '.txt'))
        with open(txt_file, 'w') as f:
            m = 0
            for id in range(start, end):
                m += 1
                f.write('\n' + img_list[id])
        print(f'Already save label.txt in {txt_file}.')
        start = end

    with open(
            os.path.join(os.path.dirname(args.dataset_path), 'label.txt'),
            'w') as f:
        f.write('\n'.join(label_list))
    print(
        f'Already save label.txt in {os.path.abspath(os.path.join(os.path.dirname(args.dataset_path), "label.txt"))}.'
    )


if __name__ == '__main__':
    args = parse_args()
    main(args)
