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
import numpy as np

from paddle_serving_client import Client
from paddle_serving_app.reader import *
import cv2

preprocess = DetectionSequential([
    DetectionFile2Image(),
    DetectionNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True),
    DetectionResize(
        (640, 640), False, interpolation=2), DetectionTranspose((2, 0, 1))
])
postprocess = RCNNPostprocess("label_list.txt", "output")

client = Client()
client.load_client_config(
    "picodet_PPLCNet_x2_5_mainbody_lite_v1.0_client/serving_client_conf.prototxt"
)
client.connect(['127.0.0.1:9293'])

im, im_info = preprocess(sys.argv[1])
im_shape = np.array(im.shape[1:]).reshape(-1)
scale_factor = np.array(list(im_info['scale_factor'])).reshape(-1)
fetch_map = client.predict(
    feed={
        "image": im,
        "im_shape": im_shape,
        "scale_factor": scale_factor,
    },
    fetch=["save_infer_model/scale_0.tmp_1"],
    batch=False)
fetch_map["image"] = sys.argv[1]
postprocess(fetch_map)
