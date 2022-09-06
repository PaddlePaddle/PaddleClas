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
import json
import os

import requests

image_path = "../../drink_dataset_v2.0/test_images/100.jpeg"


def bytes_to_base64(image_bytes: bytes) -> bytes:
    """encode bytes using base64 algorithm

    Args:
        image_bytes (bytes): bytes object to be encoded

    Returns:
        bytes: base64 bytes
    """
    return base64.b64encode(image_bytes).decode('utf8')


if __name__ == "__main__":
    url = "http://127.0.0.1:18081/recognition/prediction"

    with open(os.path.join(".", image_path), 'rb') as file:
        image_bytes = file.read()

    image_base64 = bytes_to_base64(image_bytes)
    data = {"key": ["image"], "value": [image_base64]}

    for i in range(1):
        r = requests.post(url=url, data=json.dumps(data))
        print(r.json())
