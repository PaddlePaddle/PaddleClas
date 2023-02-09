# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved. 
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
from pathlib import Path
from ..cls_task import ClsRunner


class ShiTuRunner(ClsRunner):
    def predict(self, config_file_path, device):
        self.distributed(device)
        cmd = f"{self.python} tools/infer.py -c {config_file_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_file_path, device):
        # `device` unused
        cmd = f"{self.python} tools/export_model.py -c {config_file_path}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_file_path, device):
        self.distributed(device)
        cmd = f"{self.python} python/predict_cls.py -c {config_file_path}"
        self.run_cmd(cmd, switch_wdir='deploy', echo=True, silent=False)

    def compression(self, config_file_path, device):
        self.train(config_file_path, device)
