// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#ifdef WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include "include/utility.h"
#include "yaml-cpp/yaml.h"
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <vector>

namespace PaddleClas {

    class ClsConfig {
    public:
        explicit ClsConfig(const std::string &path) {
            ReadYamlConfig(path);
            this->infer_imgs =
                    this->config_file["Global"]["infer_imgs"].as<std::string>();
            this->batch_size = this->config_file["Global"]["batch_size"].as<int>();
            this->use_gpu = this->config_file["Global"]["use_gpu"].as<bool>();
            if (this->config_file["Global"]["gpu_id"].IsDefined())
                this->gpu_id = this->config_file["Global"]["gpu_id"].as<int>();
            else
                this->gpu_id = 0;
            this->gpu_mem = this->config_file["Global"]["gpu_mem"].as<int>();
            this->cpu_threads =
                    this->config_file["Global"]["cpu_num_threads"].as<int>();
            this->use_mkldnn = this->config_file["Global"]["enable_mkldnn"].as<bool>();
            this->use_tensorrt = this->config_file["Global"]["use_tensorrt"].as<bool>();
            this->use_fp16 = this->config_file["Global"]["use_fp16"].as<bool>();
            this->enable_benchmark =
                    this->config_file["Global"]["enable_benchmark"].as<bool>();
            this->ir_optim = this->config_file["Global"]["ir_optim"].as<bool>();
            this->enable_profile =
                    this->config_file["Global"]["enable_profile"].as<bool>();
            this->cls_model_path =
                    this->config_file["Global"]["inference_model_dir"].as<std::string>() +
                    OS_PATH_SEP + "inference.pdmodel";
            this->cls_params_path =
                    this->config_file["Global"]["inference_model_dir"].as<std::string>() +
                    OS_PATH_SEP + "inference.pdiparams";
            this->resize_short_size =
                    this->config_file["PreProcess"]["transform_ops"][0]["ResizeImage"]
                    ["resize_short"]
                            .as<int>();
            this->crop_size =
                    this->config_file["PreProcess"]["transform_ops"][1]["CropImage"]["size"]
                            .as<int>();
            this->scale = this->config_file["PreProcess"]["transform_ops"][2]
            ["NormalizeImage"]["scale"]
                    .as<float>();
            this->mean = this->config_file["PreProcess"]["transform_ops"][2]
                         ["NormalizeImage"]["mean"]
                                 .as < std::vector < float >> ();
            this->std = this->config_file["PreProcess"]["transform_ops"][2]
                        ["NormalizeImage"]["std"]
                                .as < std::vector < float >> ();
            if (this->config_file["Global"]["benchmark"].IsDefined())
                this->benchmark = this->config_file["Global"]["benchmark"].as<bool>();
            else
                this->benchmark = false;

            if (this->config_file["PostProcess"]["Topk"]["topk"].IsDefined())
                this->topk = this->config_file["PostProcess"]["Topk"]["topk"].as<int>();
            if (this->config_file["PostProcess"]["Topk"]["class_id_map_file"]
                    .IsDefined())
                this->class_id_map_path =
                        this->config_file["PostProcess"]["Topk"]["class_id_map_file"]
                                .as<std::string>();
            if (this->config_file["PostProcess"]["SavePreLabel"]["save_dir"]
                    .IsDefined())
                this->label_save_dir =
                        this->config_file["PostProcess"]["SavePreLabel"]["save_dir"]
                                .as<std::string>();
            ReadLabelMap();
        }

        YAML::Node config_file;
        bool use_gpu = false;
        int gpu_id = 0;
        int gpu_mem = 4000;
        int cpu_threads = 1;
        bool use_mkldnn = false;
        bool use_tensorrt = false;
        bool use_fp16 = false;
        bool benchmark = false;
        int batch_size = 1;
        bool enable_benchmark = false;
        bool ir_optim = true;
        bool enable_profile = false;

        std::string cls_model_path;
        std::string cls_params_path;
        std::string infer_imgs;

        int resize_short_size = 256;
        int crop_size = 224;
        float scale = 0.00392157;
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};
        int topk = 5;
        std::string class_id_map_path;
        std::map<int, std::string> id_map;
        std::string label_save_dir;

        void PrintConfigInfo();

        void ReadLabelMap();

        void ReadYamlConfig(const std::string &path);
    };
} // namespace PaddleClas
