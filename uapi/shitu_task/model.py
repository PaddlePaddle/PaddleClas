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

from ..base.utils.path import get_cache_dir
from ..base.utils.arg import CLIArgument
from ..base.utils.misc import abspath
from ..cls_task import ClsModel
from .config import ShiTuConfig


class ShiTuModel(ClsModel):
    def update_config_cls(self):
        self.config_cls = ShiTuConfig

    def predict(self,
                weight_path=None,
                device=None,
                input_path=None,
                save_dir=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        if weight_path is not None:
            config.update([f'Global.pretrained_model={weight_path.replace(".pdparams","")}'])
        if input_path is not None:
            config.update([f'Infer.infer_imgs={input_path}'])
        if save_dir is not None:
            pass
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.predict(config_file_path, device)

    def export(self, weight_path=None, save_dir=None, input_shape=None):
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['config_path']
        config = self.config_cls.build_from_file(config_file_path)
        if weight_path is not None:
            config.update([f'Global.pretrained_model={weight_path.replace(".pdparams","")}'])
        if save_dir is not None:
            config.update([f'Global.save_inference_dir={save_dir}'])
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.export(config_file_path, None)

    def infer(self, model_dir, device=None, input_path=None, save_dir=None):
        model_dir = abspath(model_dir)
        if input_path is not None:
            input_path = abspath(input_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)
        
        config_file_path = '../deploy/configs/inference_cls.yaml'
        config = self.config_cls.build_from_file(config_file_path)
        config.update([f'Global.inference_model_dir={model_dir}'])
        if input_path is not None:
            config.update([f'Global.infer_imgs={input_path}'])
        if device is not None:
            config.update([f'Global.use_gpu={device.split(":")[0]=="gpu"}'])
        
        config_file_path = self.config_file_path
        config.dump(config_file_path)
        self.runner.infer(config_file_path, device)

    def compression(self,
                    dataset,
                    batch_size=None,
                    learning_rate=None,
                    epochs_iters=None,
                    device=None,
                    weight_path=None,
                    save_dir=None):
        dataset = abspath(dataset)
        if weight_path is not None:
            weight_path = abspath(weight_path)
        if save_dir is not None:
            save_dir = abspath(save_dir)

        # Update YAML config file
        config_file_path = self.arch_info['auto_compression_config_path']
        config = self.config_cls.build_from_file(config_file_path)
        config._update_dataset_config(dataset)
        config._update_device_config(device)

        if batch_size is not None:
            config._update_batch_size_config(batch_size)
        if learning_rate is not None:
            config._update_lr_config(learning_rate)
        if epochs_iters is not None:
            config.update([f'Global.epochs={epochs_iters}'])
        if weight_path is not None:
            config.update([f'Global.pretrained_model={weight_path.replace(".pdparams","")}'])
        if save_dir is not None:
            config.update([f'Global.output_dir={save_dir}'])
        config_file_path = self.config_file_path
        config.dump(config_file_path)

        self.runner.compression(config_file_path, device)
