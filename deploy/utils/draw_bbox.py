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
from PIL import Image, ImageDraw, ImageFont


def draw_bbox_results(image,
                      results,
                      input_path,
                      font_path="./utils/simfang.ttf",
                      save_dir=None):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    font_size = 18
    font = ImageFont.truetype(font_path, font_size, encoding="utf-8")

    color = (0, 102, 255)

    for result in results:
        # empty results
        if result["rec_docs"] is None:
            continue

        xmin, ymin, xmax, ymax = result["bbox"]
        text = "{}, {:.2f}".format(result["rec_docs"], result["rec_scores"])
        th = font_size
        tw = font.getlength(text)
        # tw = int(len(result["rec_docs"]) * font_size) + 60
        start_y = max(0, ymin - th)

        draw.rectangle(
            [(xmin + 1, start_y), (xmin + tw + 1, start_y + th)], fill=color)

        draw.text((xmin + 1, start_y), text, fill=(255, 255, 255), font=font)

        draw.rectangle(
            [(xmin, ymin), (xmax, ymax)], outline=(255, 0, 0), width=2)

    image_name = os.path.basename(input_path)
    if save_dir is None:
        save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, image_name)

    image.save(output_path, quality=95)
    return np.array(image)
