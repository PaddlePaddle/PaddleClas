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

import paddlehub as hub

image_path = ["./deploy/hubserving/ILSVRC2012_val_00006666.JPEG", ]
top_k = 5
module = hub.Module(name="clas_system")
res = module.predict(paths=image_path, top_k=top_k)
for i, image in enumerate(image_path):
    print("The returned result of {}: {}".format(image, res[i]))
