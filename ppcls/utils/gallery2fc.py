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

import paddle
import cv2

import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../')))

from ppcls.arch import build_model
from ppcls.utils.config import parse_config, parse_args
from ppcls.utils.save_load import load_dygraph_pretrain
from ppcls.utils.logger import init_logger
from ppcls.data import create_operators
from ppcls.arch.slim import quantize_model


class GalleryLayer(paddle.nn.Layer):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        embedding_size = self.configs["Arch"]["Head"]["embedding_size"]
        self.batch_size = self.configs["IndexProcess"]["batch_size"]
        self.image_shape = self.configs["Global"]["image_shape"].copy()
        self.image_shape.insert(0, self.batch_size)

        image_root = self.configs["IndexProcess"]["image_root"]
        data_file = self.configs["IndexProcess"]["data_file"]
        delimiter = self.configs["IndexProcess"]["delimiter"]
        self.gallery_images = []
        gallery_docs = []
        gallery_labels = []

        with open(data_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for ori_line in lines:
                line = ori_line.strip().split(delimiter)
                text_num = len(line)
                assert text_num >= 2, f"line({ori_line}) must be splitted into at least 2 parts, but got {text_num}"
                image_file = os.path.join(image_root, line[0])

                self.gallery_images.append(image_file)
                gallery_docs.append(ori_line.strip())
                gallery_labels.append(line[1].strip())
        self.gallery_layer = paddle.nn.Linear(
            embedding_size, len(self.gallery_images), bias_attr=False)
        self.gallery_layer.skip_quant = True
        output_label_str = ""
        for i, label_i in enumerate(gallery_labels):
            output_label_str += "{} {}\n".format(i, label_i)
        output_path = configs["Global"]["save_inference_dir"] + "_label.txt"

        save_dir = os.path.dirname(configs["Global"]["save_inference_dir"])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(output_path, "w") as f:
            f.write(output_label_str)

    def forward(self, x, label=None):
        x = paddle.nn.functional.normalize(x)
        x = self.gallery_layer(x)
        return x

    def build_gallery_layer(self, feature_extractor):
        transform_configs = self.configs["IndexProcess"]["transform_ops"]
        preprocess_ops = create_operators(transform_configs)
        embedding_size = self.configs["Arch"]["Head"]["embedding_size"]
        batch_index = 0
        input_tensor = paddle.zeros(self.image_shape)
        gallery_feature = paddle.zeros(
            (len(self.gallery_images), embedding_size))
        for i, image_path in enumerate(self.gallery_images):
            image = cv2.imread(image_path)[:, :, ::-1]
            for op in preprocess_ops:
                image = op(image)
            input_tensor[batch_index] = image
            batch_index += 1
            if batch_index == self.batch_size or i == len(
                    self.gallery_images) - 1:
                batch_feature = feature_extractor(input_tensor)["features"]
                for j in range(batch_index):
                    feature = batch_feature[j]
                    norm_feature = paddle.nn.functional.normalize(
                        feature, axis=0)
                    gallery_feature[i - batch_index + j + 1] = norm_feature
                batch_index = 0
        self.gallery_layer.set_state_dict({"_layer.weight": gallery_feature.T})


def export_fuse_model(configs):
    slim_config = configs["Slim"].copy()
    configs["Slim"] = None
    fuse_model = build_model(configs)
    fuse_model.head = GalleryLayer(configs)
    configs["Slim"] = slim_config
    quantize_model(configs, fuse_model)
    load_dygraph_pretrain(fuse_model, configs["Global"]["pretrained_model"])
    fuse_model.eval()
    fuse_model.head.build_gallery_layer(fuse_model)
    save_path = configs["Global"]["save_inference_dir"]
    fuse_model.quanter.save_quantized_model(
        fuse_model,
        save_path,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None] + configs["Global"]["image_shape"],
                dtype='float32')
        ])


def main():
    args = parse_args()
    configs = parse_config(args.config)
    init_logger(name='gallery2fc')
    export_fuse_model(configs)


if __name__ == '__main__':
    main()
