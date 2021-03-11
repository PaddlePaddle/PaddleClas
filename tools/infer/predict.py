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

import numpy as np
import cv2
import time

import sys
sys.path.insert(0, ".")
import tools.infer.utils as utils
from tools.infer.utils import get_image_list


class Predictor(object):
    def __init__(self, args):
        if args.enable_benchmark:
            assert args.model is not None
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args

        self.paddle_predictor = utils.create_paddle_predictor(args)
        input_names = self.paddle_predictor.get_input_names()
        self.input_tensor = self.paddle_predictor.get_input_handle(input_names[
            0])

        output_names = self.paddle_predictor.get_output_names()
        self.output_tensor = self.paddle_predictor.get_output_handle(
            output_names[0])

    def predict(self, batch_input):
        self.input_tensor.copy_from_cpu(batch_input)
        self.paddle_predictor.run()
        batch_output = self.output_tensor.copy_to_cpu()
        return batch_output

    def local_predict(self):
        image_list = get_image_list(self.args.image_file)
        batch_input_list = []
        filepath_list = []
        for idx, img_path in enumerate(image_list):
            img = cv2.imread(img_path)[:, :, ::-1]
            assert img is not None, "Error in loading image: {}".format(
                img_path)
            img = utils.preprocess(img, args)
            batch_input_list.append(img)
            filepath_list.append(img_path)

            if (idx + 1) % args.batch_size == 0 or (idx + 1
                                                    ) == len(filepath_list):
                batch_outputs = self.predict(np.array(batch_input_list))
                batch_result_list = utils.postprocess(batch_outputs,
                                                      self.args.top_k)

                for number, result_list in enumerate(batch_result_list):
                    filename = filepath_list[number].split("/")[-1]
                    print("File:{}, The top-{} result(s):{}".format(
                        filename, self.args.top_k, result_list))
                batch_input_list = []
                filepath_list = []

    def benchmark_predict(self):
        test_num = 500
        test_time = 0.0
        for i in range(0, test_num + 10):
            inputs = np.random.rand(args.batch_size, 3, 224,
                                    224).astype(np.float32)
            start_time = time.time()
            batch_output = self.predict(inputs).flatten()
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
    predictor = Predictor(args)
    if not args.enable_benchmark:
        predictor.local_predict()
    else:
        predictor.benchmark_predict()
