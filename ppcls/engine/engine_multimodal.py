# copyright (c) 2023 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import platform
import paddle
import paddle.distributed as dist
from visualdl import LogWriter
from paddle import nn
import numpy as np
import random

from ppcls.utils import logger
from ppcls.utils import save_predict_result

from ppcls.data.utils.get_image_list import get_image_list
from ppcls.engine.train.utils import type_name
from ppcls.engine import engine
from ppcls.engine import train as train_method
from ppcls.engine.engine import ExportModel, load_dygraph_pretrain

def collect_fn_list(data):
    image_ram_list = []
    caption_list = []
    image_tag_list = []
    image_parse_tag_list = []
    image_clip_list = []
    for item in data:
        i1,i2,i3,i4,i5 = item
        image_ram_list.append(i1)
        caption_list.append(i2)
        image_tag_list.append(i3)
        image_parse_tag_list.append(i4)
        image_clip_list.append(i5)
    
    image_rams = paddle.stack(image_ram_list)
    image_tags = paddle.stack(image_tag_list)
    image_parse_tags = paddle.stack(image_parse_tag_list)
    image_clips = paddle.stack(image_clip_list)
    return (image_rams, caption_list , image_tags, image_parse_tags, image_clips)

class EngineMultimodal(engine.Engine):
    def __init__(self, config, mode="train"):
        super().__init__(config, mode)
        self.train_epoch_func = train_method.train_epoch_multimodal
        self.train_dataloader.collate_fn = collect_fn_list
        self.eval_dataloader.collate_fn = collect_fn_list

    @paddle.no_grad()
    def eval(self, epoch_id=0):
        assert self.mode in ["train", "eval"]
        self.model.eval()
        eval_result = self.eval_func(self, epoch_id)
        self.model.train()
        return eval_result

    @paddle.no_grad()
    def infer(self):
        assert self.mode == "infer" and self.eval_mode == "classification"
        results = []
        total_trainer = dist.get_world_size()
        local_rank = dist.get_rank()
        infer_imgs = self.config["Infer"]["infer_imgs"]
        infer_list = self.config["Infer"].get("infer_list", None)
        image_list = get_image_list(infer_imgs, infer_list=infer_list)
        # data split
        image_list = image_list[local_rank::total_trainer]

        batch_size = self.config["Infer"]["batch_size"]
        self.model.eval()
        batch_data = []
        image_file_list = []
        save_path = self.config["Infer"].get("save_dir", None)
        for idx, image_file in enumerate(image_list):
            with open(image_file, 'rb') as f:
                x = f.read()
            for process in self.preprocess_func:
                x = process(x)
            batch_data.append(x)
            image_file_list.append(image_file)
            if len(batch_data) >= batch_size or idx == len(image_list) - 1:
                batch_tensor = paddle.to_tensor(batch_data)

                with self.auto_cast(is_eval=True):
                    tag_output = self.model.inference(batch_tensor)

                result = self.postprocess_func(*tag_output, image_file_list)
                if not save_path:
                    logger.info(result)
                results.extend(result)
                batch_data.clear()
                image_file_list.clear()
        if save_path:
            save_predict_result(save_path, results)
        return results

    def export(self):
        assert self.mode == "export"
        use_multilabel = self.config["Global"].get("use_multilabel", False)
        model = ExportModelMultiModal(self.config["Arch"], self.model,
                                      use_multilabel)
        if self.config["Global"]["pretrained_model"] is not None:
            load_dygraph_pretrain(model.base_model,
                                  self.config["Global"]["pretrained_model"])

        model.eval()

        # for re-parameterization nets
        for layer in self.model.sublayers():
            if hasattr(layer, "re_parameterize") and not getattr(layer,
                                                                 "is_repped"):
                layer.re_parameterize()

        save_path = os.path.join(self.config["Global"]["save_inference_dir"],
                                 "inference")

        model = paddle.jit.to_static(
            model,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None] + self.config["Global"]["image_shape"],
                    dtype='float32')
            ])
        if hasattr(model.base_model,
                   "quanter") and model.base_model.quanter is not None:
            model.base_model.quanter.save_quantized_model(model,
                                                          save_path + "_int8")
        else:
            paddle.jit.save(model, save_path)
        if self.config["Global"].get("export_for_fd", False):
            src_path = self.config["Global"]["infer_config_path"]
            dst_path = os.path.join(
                self.config["Global"]["save_inference_dir"], 'inference.yml')
            shutil.copy(src_path, dst_path)
        logger.info(
            f"Export succeeded! The inference model exported has been saved in \"{self.config['Global']['save_inference_dir']}\"."
        )


class ExportModelMultiModal(ExportModel):
    def forward(self, x):
        x = self.base_model.inference(x)
        if isinstance(x, list):
            x = x[0]
        if self.infer_model_name is not None:
            x = x[self.infer_model_name]
        if self.infer_output_key is not None:
            x = x[self.infer_output_key]
        return x
