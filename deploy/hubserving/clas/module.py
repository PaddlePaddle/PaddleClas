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
sys.path.insert(0, ".")

import time

from paddlehub.common.logger import logger
from paddlehub.module.module import moduleinfo, serving
import cv2
import numpy as np
import paddlehub as hub

import tools.infer.predict as paddle_predict
from tools.infer.utils import Base64ToCV2
from deploy.hubserving.clas.params import read_params


@moduleinfo(
    name="clas_system",
    version="1.0.0",
    summary="class system service",
    author="paddle-dev",
    author_email="paddle-dev@baidu.com",
    type="cv/class")
class ClasSystem(hub.Module):
    def _initialize(self, use_gpu=None):
        """
        initialize with the necessary elements
        """
        cfg = read_params()
        if use_gpu is not None:
            cfg.use_gpu = use_gpu
        cfg.hubserving = True
        cfg.enable_benchmark = False
        self.args = cfg
        if cfg.use_gpu:
            try:
                _places = os.environ["CUDA_VISIBLE_DEVICES"]
                int(_places[0])
                print("Use GPU, GPU Memery:{}".format(cfg.gpu_mem))
                print("CUDA_VISIBLE_DEVICES: ", _places)
            except:
                raise RuntimeError(
                    "Environment Variable CUDA_VISIBLE_DEVICES is not set correctly. If you wanna use gpu, please set CUDA_VISIBLE_DEVICES via export CUDA_VISIBLE_DEVICES=cuda_device_id."
                )
        else:
            print("Use CPU")

    def read_images(self, paths=[]):
        images = []
        for img_path in paths:
            assert os.path.isfile(
                img_path), "The {} isn't a valid file.".format(img_path)
            img = cv2.imread(img_path)
            if img is None:
                logger.info("error in loading image:{}".format(img_path))
                continue
            img = img[:, :, ::-1]
            images.append(img)
        return images

    def predict(self, images=[], paths=[], top_k=1):
        """
        
        Args:
            images (list(numpy.ndarray)): images data, shape of each is [H, W, C]. If images not paths
            paths (list[str]): The paths of images. If paths not images
        Returns:
            res (list): The result of chinese texts and save path of images.
        """

        if images != [] and isinstance(images, list) and paths == []:
            predicted_data = images
        elif images == [] and isinstance(paths, list) and paths != []:
            predicted_data = self.read_images(paths)
        else:
            raise TypeError(
                "The input data is inconsistent with expectations.")

        assert predicted_data != [], "There is not any image to be predicted. Please check the input data."

        all_results = []
        for img in predicted_data:
            if img is None:
                logger.info("error in loading image")
                all_results.append([])
                continue
            starttime = time.time()

            self.args.image_file = img
            self.args.top_k = top_k
            classes, scores = paddle_predict.main(self.args)

            elapse = time.time() - starttime
            logger.info("Predict time: {}".format(elapse))
            all_results.append([classes.tolist(), scores.tolist()])

        return all_results

    @serving
    def serving_method(self, images, **kwargs):
        """
        Run as a service.
        """
        to_cv2 = Base64ToCV2()
        images_decode = [to_cv2(image) for image in images]
        results = self.predict(images_decode, **kwargs)
        return results


if __name__ == '__main__':
    clas = ClasSystem()
    image_path = ['./deploy/hubserving/ILSVRC2012_val_00006666.JPEG', ]
    res = clas.predict(paths=image_path, top_k=5)
    print(res)
