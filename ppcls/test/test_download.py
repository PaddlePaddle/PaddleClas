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

import unittest
import dl
import os
import shutil

url = "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_pretrained.tar"


class DownloadDecompressTestCase(unittest.TestCase):
    def setUp(self):
        print("Test Download and Decompress Function...")

    def test_decompress(self):
        if os.path.exists('./ResNet50_vd_pretrained'):
            shutil.rmtree('./ResNet50_vd_pretrained')
        if os.path.exists("./ResNet50_vd_pretrained.tar"):
            shutil.rmtree("./ResNet50_vd_pretrained.tar")

        dl.decompress(dl.download(url, "./"))
        self.assertTrue(os.path.exists("./ResNet50_vd_pretrained"))
        shutil.rmtree('./ResNet50_vd_pretrained')


if __name__ == "__main__":
    unittest.main()
