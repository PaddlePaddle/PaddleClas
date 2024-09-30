# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
import yaml
import paddle

from . import logger


def save_predict_result(save_path, result):
    if os.path.splitext(save_path)[-1] == '':
        if save_path[-1] == "/":
            save_path = save_path[:-1]
        save_path = save_path + '.json'
    elif os.path.splitext(save_path)[-1] == '.json':
        save_path = save_path
    else:
        raise Exception(
            f"{save_path} is invalid input path, only files in json format are supported."
        )

    if os.path.exists(save_path):
        logger.warning(f"The file {save_path} will be overwritten.")
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(result, f)


def update_train_results(config,
                         prefix,
                         metric_info,
                         done_flag=False,
                         last_num=5,
                         ema=False):

    if paddle.distributed.get_rank() != 0:
        return

    assert last_num >= 1
    train_results_path = os.path.join(config["Global"]["output_dir"],
                                      "train_results.json")
    save_model_tag = ["pdparams", "pdopt", "pdstates"]
    save_inference_tag = [
        "inference_config", "pdmodel", "pdiparams", "pdiparams.info"
    ]
    if ema:
        save_model_tag.append("pdema")
    if os.path.exists(train_results_path):
        with open(train_results_path, "r") as fp:
            train_results = json.load(fp)
    else:
        train_results = {}
        train_results["model_name"] = config["Global"].get("pdx_model_name",
                                                           None)
        if config.get("infer", None):
            train_results["label_dict"] = config["Infer"]["PostProcess"].get(
                "class_id_map_file", "")
        else:
            train_results["label_dict"] = ""
        train_results["train_log"] = "train.log"
        train_results["visualdl_log"] = ""
        train_results["config"] = "config.yaml"
        train_results["models"] = {}
        for i in range(1, last_num + 1):
            train_results["models"][f"last_{i}"] = {}
        train_results["models"]["best"] = {}
    train_results["done_flag"] = done_flag
    if prefix == "best_model":
        train_results["models"]["best"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, f"{prefix}.{tag}")
        for tag in save_inference_tag:
            train_results["models"]["best"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")
    else:
        for i in range(last_num - 1, 0, -1):
            train_results["models"][f"last_{i + 1}"] = train_results["models"][
                f"last_{i}"].copy()
        train_results["models"][f"last_{1}"]["score"] = metric_info["metric"]
        for tag in save_model_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, f"{prefix}.{tag}")
        for tag in save_inference_tag:
            train_results["models"][f"last_{1}"][tag] = os.path.join(
                prefix, "inference", f"inference.{tag}"
                if tag != "inference_config" else "inference.yml")

    with open(train_results_path, "w") as fp:
        json.dump(train_results, fp)
