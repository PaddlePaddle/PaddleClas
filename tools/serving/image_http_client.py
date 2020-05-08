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

import requests
import base64
import json
import sys
import numpy as np

py_version = sys.version_info[0]


def predict(image_path, server):
    if py_version == 2:
        image = base64.b64encode(open(image_path).read())
    else:
        image = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
    req = json.dumps({"feed": [{"image": image}], "fetch": ["prediction"]})
    r = requests.post(
        server, data=req, headers={"Content-Type": "application/json"})
    try:
        pred = r.json()["result"]["prediction"][0]
        cls_id = np.argmax(pred)
        score = pred[cls_id]
        pred = {"cls_id": cls_id, "score": score}
        return pred
    except ValueError:
        print(r.text)
    return r


if __name__ == "__main__":
    server = "http://127.0.0.1:{}/image/prediction".format(sys.argv[1])
    image_file = sys.argv[2]
    res = predict(image_file, server)
    print("res:", res)
