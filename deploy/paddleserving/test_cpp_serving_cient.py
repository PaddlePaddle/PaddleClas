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

#app
from paddle_serving_app.reader import Sequential, URL2Image, Resize
from paddle_serving_app.reader import CenterCrop, RGB2BGR, Transpose, Div, Normalize
import time

client = Client()
client.load_client_config("./ResNet50_vd_serving/serving_server_conf.prototxt")
client.connect(["127.0.0.1:9292"])

label_dict = {}
label_idx = 0
with open("imagenet.label") as fin:
    for line in fin:
        label_dict[label_idx] = line.strip()
        label_idx += 1

#preprocess
seq = Sequential([
    URL2Image(), Resize(256), CenterCrop(224), RGB2BGR(), Transpose((2, 0, 1)),
    Div(255), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], True)
])

start = time.time()
image_file = "https://paddle-serving.bj.bcebos.com/imagenet-example/daisy.jpg"
for i in range(1):
    img = seq(image_file)
    fetch_map = client.predict(
        feed={"inputs": img}, fetch=["prediction"], batch=False)

    prob = max(fetch_map["prediction"][0])
    label = label_dict[fetch_map["prediction"][0].tolist().index(prob)].strip(
    ).replace(",", "")
    print("prediction: {}, probability: {}".format(label, prob))
end = time.time()
print(end - start)
