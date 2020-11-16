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

import sys
sys.path.insert(0, ".")
import tools.infer.utils as utils
import numpy as np
import cv2
import time

from paddle.inference import Config
from paddle.inference import create_predictor


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)

    if args.use_gpu:
        assert args.enable_mkldnn is False, "Error: Cannot use GPU and MKL-DNN at the same time"
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()

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


def main(args):
    if not args.enable_benchmark:
        assert args.batch_size == 1
        assert args.use_fp16 is False
    else:
        assert args.use_gpu is True
        assert args.model is not None
    # HALF precission predict only work when using tensorrt
    if args.use_fp16 is True:
        assert args.use_tensorrt is True

    predictor = create_paddle_predictor(args)

    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])

    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])

    test_num = 500
    test_time = 0.0
    if not args.enable_benchmark:
        # for PaddleHubServing
        if args.hubserving:
            img = args.image_file
        # for predict only
        else:
            img = cv2.imread(args.image_file)[:, :, ::-1]
        assert img is not None, "Error in loading image: {}".format(
            args.image_file)
        inputs = utils.preprocess(img, args)
        inputs = np.expand_dims(
            inputs, axis=0).repeat(
                args.batch_size, axis=0).copy()
        input_tensor.copy_from_cpu(inputs)

        predictor.run()

        output = output_tensor.copy_to_cpu()
        return utils.postprocess(output, args)
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
            args.model, trt_msg, fp_message, args.batch_size, 1000 * test_time
            / test_num))


if __name__ == "__main__":
    args = utils.parse_args()
    classes, scores = main(args)
    print("Current image file: {}".format(args.image_file))
    print("\ttop-1 class: {0}".format(classes[0]))
    print("\ttop-1 score: {0}".format(scores[0]))
