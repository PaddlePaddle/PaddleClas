// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_inference_api.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <cstring>
#include <fstream>
#include <numeric>

#include <include/preprocess_op.h>
#include <include/yaml_config.h>

using namespace paddle_infer;

namespace Feature {

    class FeatureExtracter {
    public:
        explicit FeatureExtracter(const YAML::Node &config_file) {
            this->use_gpu_ = config_file["Global"]["use_gpu"].as<bool>();
            if (config_file["Global"]["gpu_id"].IsDefined())
                this->gpu_id_ = config_file["Global"]["gpu_id"].as<int>();
            else
                this->gpu_id_ = 0;
            this->gpu_mem_ = config_file["Global"]["gpu_mem"].as<int>();
            this->cpu_math_library_num_threads_ =
                    config_file["Global"]["cpu_num_threads"].as<int>();
            this->use_mkldnn_ = config_file["Global"]["enable_mkldnn"].as<bool>();
            this->use_tensorrt_ = config_file["Global"]["use_tensorrt"].as<bool>();
            this->use_fp16_ = config_file["Global"]["use_fp16"].as<bool>();

            this->cls_model_path_ =
                    config_file["Global"]["rec_inference_model_dir"].as<std::string>() +
                    OS_PATH_SEP + "inference.pdmodel";
            this->cls_params_path_ =
                    config_file["Global"]["rec_inference_model_dir"].as<std::string>() +
                    OS_PATH_SEP + "inference.pdiparams";
            this->resize_size_ =
                    config_file["RecPreProcess"]["transform_ops"][0]["ResizeImage"]["size"]
                            .as<int>();
            this->scale_ = config_file["RecPreProcess"]["transform_ops"][1]["NormalizeImage"]["scale"].as<float>();
            this->mean_ = config_file["RecPreProcess"]["transform_ops"][1]
                          ["NormalizeImage"]["mean"]
                                  .as < std::vector < float >> ();
            this->std_ = config_file["RecPreProcess"]["transform_ops"][1]
                         ["NormalizeImage"]["std"]
                                 .as < std::vector < float >> ();
            if (config_file["Global"]["rec_feature_normlize"].IsDefined())
                this->feature_norm =
                        config_file["Global"]["rec_feature_normlize"].as<bool>();

            LoadModel(cls_model_path_, cls_params_path_);
        }

        // Load Paddle inference model
        void LoadModel(const std::string &model_path, const std::string &params_path);

        // Run predictor
        void Run(cv::Mat &img, std::vector<float> &out_data,
                 std::vector<double> &times);

        void FeatureNorm(std::vector<float> &feature);

        std::shared_ptr <Predictor> predictor_;

    private:
        bool use_gpu_ = false;
        int gpu_id_ = 0;
        int gpu_mem_ = 4000;
        int cpu_math_library_num_threads_ = 4;
        bool use_mkldnn_ = false;
        bool use_tensorrt_ = false;
        bool feature_norm = true;
        bool use_fp16_ = false;
        std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
        std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
        float scale_ = 0.00392157;
        int resize_size_ = 224;
        int resize_short_ = 224;
        std::string cls_model_path_;
        std::string cls_params_path_;

        // pre-process
        ResizeImg resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;
    };

} // namespace Feature
