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

import base64
import time

from paddle_serving_client import Client


def bytes_to_base64(image: bytes) -> str:
    """encode bytes into base64 string
    """
    return base64.b64encode(image).decode('utf8')


client = Client()
client.load_client_config("./ResNet50_client/serving_client_conf.prototxt")
client.connect(["127.0.0.1:9292"])

label_dict = {}
label_idx = 0
with open("imagenet.label") as fin:
    for line in fin:
        label_dict[label_idx] = line.strip()
        label_idx += 1

image_file = "./daisy.jpg"
for i in range(1):
    start = time.time()
    with open(image_file, 'rb') as img_file:
        image_data = img_file.read()
        image = bytes_to_base64(image_data)
        fetch_dict = client.predict(
            feed={"inputs": image}, fetch=["prediction"], batch=False)
        prob = max(fetch_dict["prediction"][0])
        label = label_dict[fetch_dict["prediction"][0].tolist().index(
            prob)].strip().replace(",", "")
        print("prediction: {}, probability: {}".format(label, prob))
    end = time.time()
    print(end - start)
