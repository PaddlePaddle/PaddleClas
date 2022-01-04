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

#include "include/cls_config.h"
#include <include/preprocess_op.h>

using namespace paddle_infer;

namespace PaddleClas {

    class Classifier {
    public:
        explicit Classifier(const ClsConfig &config) {
            this->use_gpu_ = config.use_gpu;
            this->gpu_id_ = config.gpu_id;
            this->gpu_mem_ = config.gpu_mem;
            this->cpu_math_library_num_threads_ = config.cpu_threads;
            this->use_fp16_ = config.use_fp16;
            this->use_mkldnn_ = config.use_mkldnn;
            this->use_tensorrt_ = config.use_tensorrt;
            this->mean_ = config.mean;
            this->std_ = config.std;
            this->resize_short_size_ = config.resize_short_size;
            this->scale_ = config.scale;
            this->crop_size_ = config.crop_size;
            this->ir_optim_ = config.ir_optim;
            LoadModel(config.cls_model_path, config.cls_params_path);
        }

        // Load Paddle inference model
        void LoadModel(const std::string &model_path, const std::string &params_path);

        // Run predictor
        void Run(cv::Mat &img, std::vector<float> &out_data, std::vector<int> &idx,
                 std::vector<double> &times);

    private:
        std::shared_ptr <Predictor> predictor_;

        bool use_gpu_ = false;
        int gpu_id_ = 0;
        int gpu_mem_ = 4000;
        int cpu_math_library_num_threads_ = 4;
        bool use_mkldnn_ = false;
        bool use_tensorrt_ = false;
        bool use_fp16_ = false;
        bool ir_optim_ = true;

        std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
        std::vector<float> std_ = {0.229f, 0.224f, 0.225f};
        float scale_ = 0.00392157;

        int resize_short_size_ = 256;
        int crop_size_ = 224;

        // pre-process
        ResizeImg resize_op_;
        Normalize normalize_op_;
        Permute permute_op_;
        CenterCropImg crop_op_;
    };

} // namespace PaddleClas
