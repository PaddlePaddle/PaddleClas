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
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../../../')))
import argparse
import numpy as np
import cv2
import paddlehub as hub
from tools.infer.utils import preprocess

args = argparse.Namespace(resize_short=256, resize=224, normalize=True)

img_path_list = ["./deploy/hubserving/ILSVRC2012_val_00006666.JPEG", ]

module = hub.Module(name="clas_system")
for i, img_path in enumerate(img_path_list):
    img = cv2.imread(img_path)[:, :, ::-1]
    img = preprocess(img, args)
    batch_input_data = np.expand_dims(img, axis=0)
    res = module.predict(batch_input_data)
    print("The returned result of {}: {}".format(img_path, res))
