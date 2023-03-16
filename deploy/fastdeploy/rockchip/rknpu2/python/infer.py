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
import fastdeploy as fd
import cv2
import os


def parse_arguments():
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file", required=True, help="Path of rknn model.")
    parser.add_argument("--config_file", required=True, help="Path of config.")
    parser.add_argument(
        "--image", type=str, required=True, help="Path of test image file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    model_file = args.model_file
    params_file = ""
    config_file = args.config_file
    # 配置runtime，加载模型
    runtime_option = fd.RuntimeOption()
    runtime_option.use_rknpu2()
    model = fd.vision.classification.ResNet50vd(
        model_file,
        params_file,
        config_file,
        runtime_option=runtime_option,
        model_format=fd.ModelFormat.RKNN)
    # 禁用通道转换    
    model.preprocessor.disable_permute()
    im = cv2.imread(args.image)
    result = model.predict(im, topk=1)
    print(result)
