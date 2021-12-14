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
from paddle_serving_client import Client

from paddle_serving_app.reader import Sequential, URL2Image, Resize
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import time

client = Client()
client.load_client_config(
    "general_PPLCNet_x2_5_lite_v1.0_client/serving_client_conf.prototxt")
client.connect(["127.0.0.1:9294"])

seq = Sequential([
    URL2Image(), Resize((224, 224)), Div(255),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], False), Transpose(
        (2, 0, 1))
])

start = time.time()
image_file = "https://paddle-serving.bj.bcebos.com/imagenet-example/daisy.jpg"
img = seq(image_file)

fetch_map = client.predict(feed={"x": img}, fetch=["features"], batch=False)
feature = fetch_map["features"]
print("feature:  {}\n  feature dim: {}".format(feature, feature.shape))
end = time.time()
print(end - start)
