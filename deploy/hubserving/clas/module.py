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

import os
import sys
sys.path.insert(0, ".")

import time

from paddlehub.utils.log import logger
from paddlehub.module.module import moduleinfo, serving
import cv2
import numpy as np
import paddle.nn as nn

from tools.infer.predict import Predictor
from tools.infer.utils import b64_to_np, postprocess
from deploy.hubserving.clas.params import read_params


@moduleinfo(
    name="clas_system",
    version="1.0.0",
    summary="class system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/class")
class ClasSystem(nn.Layer):
    def __init__(self, use_gpu=None, enable_mkldnn=None):
        """
        initialize with the necessary elements
        """
        cfg = read_params()
        if use_gpu is not None:
            cfg.use_gpu = use_gpu
        if enable_mkldnn is not None:
            cfg.enable_mkldnn = enable_mkldnn
        cfg.hubserving = True
        cfg.enable_benchmark = False
        self.args = cfg
        if cfg.use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("Use GPU, GPU Memery:{}".format(cfg.gpu_mem))
                print("CUDA_VISIBLE_DEVICES: ", _places)
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        else:
            print("Use CPU")
            print("Enable MKL-DNN") if enable_mkldnn else None
        self.predictor = Predictor(self.args)

    def predict(self, batch_input_data, top_k=1):
        assert isinstance(
            batch_input_data,
            np.ndarray), "The input data is inconsistent with expectations."

        starttime = time.time()
        batch_outputs = self.predictor.predict(batch_input_data)
        elapse = time.time() - starttime
        batch_result_list = postprocess(batch_outputs, top_k)
        return {"prediction": batch_result_list, "elapse": elapse}

    @serving
    def serving_method(self, images, revert_params, **kwargs):
        """
        Run as a service.
        """
        input_data = b64_to_np(images, revert_params)
        results = self.predict(batch_input_data=input_data, **kwargs)
        return results
