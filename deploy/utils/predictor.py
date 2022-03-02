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
import platform
import os
import argparse
import base64
import shutil
import cv2
import numpy as np

from paddle.inference import Config
from paddle.inference import create_predictor


class Predictor(object):
    def __init__(self, args, inference_model_dir=None):
        # HALF precission predict only work when using tensorrt
        if args.use_fp16 is True:
            assert args.use_tensorrt is True
        self.args = args
        if self.args.get("use_onnx", False):
            self.predictor, self.config = self.create_onnx_predictor(
                args, inference_model_dir)
        else:
            self.predictor, self.config = self.create_paddle_predictor(
                args, inference_model_dir)

    def predict(self, image):
        raise NotImplementedError

    def create_paddle_predictor(self, args, inference_model_dir=None):
        if inference_model_dir is None:
            inference_model_dir = args.inference_model_dir
        params_file = os.path.join(inference_model_dir, "inference.pdiparams")
        model_file = os.path.join(inference_model_dir, "inference.pdmodel")
        config = Config(model_file, params_file)

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # there is no set_mkldnn_cache_capatity() on macOS
                if platform.system() != "Darwin":
                    # cache 10 different shapes for mkldnn to avoid memory leak
                    config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_num_threads)

        if args.enable_profile:
            config.enable_profile()
        config.disable_glog_info()
        config.switch_ir_optim(args.ir_optim)  # default true
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=Config.Precision.Half
                if args.use_fp16 else Config.Precision.Float32,
                max_batch_size=args.batch_size,
                workspace_size=1 << 30,
                min_subgraph_size=30)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor, config

    def create_onnx_predictor(self, args, inference_model_dir=None):
        import onnxruntime as ort
        if inference_model_dir is None:
            inference_model_dir = args.inference_model_dir
        model_file = os.path.join(inference_model_dir, "inference.onnx")
        config = ort.SessionOptions()
        if args.use_gpu:
            raise ValueError(
                "onnx inference now only supports cpu! please specify use_gpu false."
            )
        else:
            config.intra_op_num_threads = args.cpu_num_threads
            if args.ir_optim:
                config.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        predictor = ort.InferenceSession(model_file, sess_options=config)
        return predictor, config
