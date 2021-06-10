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
import base64
from paddle_serving_server.web_service import WebService
import utils


class ImageService(WebService):
    def __init__(self, name):
        super(ImageService, self).__init__(name=name)
        self.operators = self.create_operators()

    def create_operators(self):
        size = 224
        img_mean = [0.485, 0.456, 0.406]
        img_std = [0.229, 0.224, 0.225]
        img_scale = 1.0 / 255.0
        decode_op = utils.DecodeImage()
        resize_op = utils.ResizeImage(resize_short=256)
        crop_op = utils.CropImage(size=(size, size))
        normalize_op = utils.NormalizeImage(
            scale=img_scale, mean=img_mean, std=img_std)
        totensor_op = utils.ToTensor()
        return [decode_op, resize_op, crop_op, normalize_op, totensor_op]

    def _process_image(self, data, ops):
        for op in ops:
            data = op(data)
        return data

    def preprocess(self, feed={}, fetch=[]):
        feed_batch = []
        for ins in feed:
            if "image" not in ins:
                raise ("feed data error!")
            sample = base64.b64decode(ins["image"])
            img = self._process_image(sample, self.operators)
            feed_batch.append({"image": img})
        return feed_batch, fetch


image_service = ImageService(name="image")
image_service.load_model_config(sys.argv[1])
image_service.prepare_server(
    workdir=sys.argv[2], port=int(sys.argv[3]), device="cpu")
image_service.run_server()
image_service.run_flask()
