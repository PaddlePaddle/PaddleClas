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

import utils
import numpy as np
import cv2
import time

from paddle.fluid.core import AnalysisConfig
from paddle.fluid.core import create_paddle_predictor


def create_predictor(args):
    config = AnalysisConfig(args.model_file, args.params_file)

    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()

    config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        config.enable_tensorrt_engine(
            precision_mode=AnalysisConfig.Precision.Half
            if args.use_fp16 else AnalysisConfig.Precision.Float32,
            max_batch_size=args.batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_paddle_predictor(config)

    return predictor


def main():
    args = utils.parse_args()

    if not args.enable_benchmark:
        assert args.batch_size == 1
        assert args.use_fp16 is False
    else:
        assert args.use_gpu is True
        assert args.model_name is not None
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    predictor = create_predictor(args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_tensor(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_tensor(output_names[0])

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        img = cv2.imread(args.image_file)[:, :, ::-1]
        inputs = utils.preprocess(img, args)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                args.batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        predictor.zero_copy_run()

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

            predictor.zero_copy_run()

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
