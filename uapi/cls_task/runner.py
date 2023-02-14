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
from ..base import BaseRunner


class ClsRunner(BaseRunner):
    def train(self, config_path, cli_args, device):
        python, _ = self.distributed(device)
        cli_args = " ".join(cli_args)
        cmd = f"{python} tools/train.py -c {config_path} {cli_args}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def predict(self, config_path, cli_args, device):
        self.distributed(device)
        cli_args = " ".join(cli_args)
        cmd = f"{self.python} tools/infer.py -c {config_path} {cli_args}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def export(self, config_path, cli_args, device):
        # `device` unused
        cli_args = " ".join(cli_args)
        cmd = f"{self.python} tools/export_model.py -c {config_path}  {cli_args}"
        self.run_cmd(cmd, switch_wdir=True, echo=True, silent=False)

    def infer(self, config_path, cli_args, device):
        self.distributed(device)
        cli_args = " ".join(cli_args)
        cmd = f"{self.python} python/predict_cls.py -c {config_path} {cli_args}"
        self.run_cmd(cmd, switch_wdir='deploy', echo=True, silent=False)

    def compression(self, config_path, cli_args, device, save_dir, model_name):
        # Step 1: Train model
        self.train(config_path, cli_args, device)

        # Step 2: Export model
        weight_path = os.path.join(save_dir, model_name, 'best_model')
        cli_args = [
            f'-o Global.pretrained_model={weight_path}',
            f'-o Global.save_inference_dir={save_dir}'
        ]
        self.export(config_path, cli_args, device)
