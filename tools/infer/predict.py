# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import utils
import numpy as np
import time

from paddle.inference import Config
from paddle.inference import create_predictor


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_file", type=str)
    parser.add_argument("-m", "--model_file", type=str)
    parser.add_argument("-p", "--params_file", type=str)
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    parser.add_argument("--use_fp16", type=str2bool, default=False)
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--gpu_mem", type=int, default=8000)
    parser.add_argument("--enable_benchmark", type=str2bool, default=False)
    parser.add_argument("--model_name", type=str)

    return parser.parse_args()


def create_paddle_predictor(args):
    print(args)
    config = Config(args.model_file, args.params_file)
    #     config = Config("./r50_rerod/r50_dyg.pdmodel", "./r50_rerod/r50_dyg.pdiparams")

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=Config.PrecisionType.Half
            if args.use_fp16 else Config.PrecisionType.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return predictor


def create_operators():
    size = 224
    img_mean = [0.485, 0.456, 0.406]
    img_std = [0.229, 0.224, 0.225]
    img_scale = 1.0 / 255.0

    decode_op = utils.DecodeImage()
    resize_op = utils.ResizeImage(resize_short=256)
    crop_op = utils.CropImage(size=(size, size))
    normalize_op = utils.NormalizeImage(
        scale=img_scale, mean=img_mean, std=img_std)
    totensor_op = utils.ToTensor()

    return [decode_op, resize_op, crop_op, normalize_op, totensor_op]


def preprocess(fname, ops):
    data = open(fname, 'rb').read()
    for op in ops:
        data = op(data)

    return data


def main():
    args = parse_args()

    if not args.enable_benchmark:
        assert args.batch_size == 1
        assert args.use_fp16 is False
    else:
        assert args.use_gpu is True
        assert args.model_name is not None
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    operators = create_operators()
    predictor = create_paddle_predictor(args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        inputs = preprocess(args.image_file, operators)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                args.batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        predictor.run()

        output = output_tensor.copy_to_cpu()
        output = output.flatten()
        cls = np.argmax(output)
        score = output[cls]
        print("Current image file: {}".format(args.image_file))
        print("\ttop-1 class: {0}".format(cls))
        print("\ttop-1 score: {0}".format(score))
    else:
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            input_tensor.copy_from_cpu(inputs)

            predictor.run()

            output = output_tensor.copy_to_cpu()
            output = output.flatten()
            if i >= 10:
                test_time += time.time() - start_time
            time.sleep(0.01)  # sleep for T4 GPU

        fp_message = "FP16" if args.use_fp16 else "FP32"
        trt_msg = "using tensorrt" if args.use_tensorrt else "not using tensorrt"
        print("{0}\t{1}\t{2}\tbatch size: {3}\ttime(ms): {4}".format(
            args.model_name, trt_msg, fp_message, args.batch_size, 1000 *
            test_time / test_num))


if __name__ == "__main__":
    main()
