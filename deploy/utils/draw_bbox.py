# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import cv2


def draw_bbox_results(image, results, input_path, save_dir=None):
    for result in results:
        [xmin, ymin, xmax, ymax] = result["bbox"]

        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                              2)

    image_name = os.path.basename(input_path)
    if save_dir is None:
        save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, image_name)
    cv2.imwrite(output_path, image)
