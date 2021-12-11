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

#include <cmath>
#include <include/feature_extracter.h>
#include <numeric>

namespace Feature {

    void FeatureExtracter::LoadModel(const std::string &model_path,
                                     const std::string &params_path) {
        paddle_infer::Config config;
        config.SetModel(model_path, params_path);

        if (this->use_gpu_) {
            config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
            if (this->use_tensorrt_) {
                config.EnableTensorRtEngine(
                        1 << 20, 1, 3,
                        this->use_fp16_ ? paddle_infer::Config::Precision::kHalf
                                        : paddle_infer::Config::Precision::kFloat32,
                        false, false);
            }
        } else {
            config.DisableGpu();
            if (this->use_mkldnn_) {
                config.EnableMKLDNN();
                // cache 10 different shapes for mkldnn to avoid memory leak
                config.SetMkldnnCacheCapacity(10);
            }
            config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
        }

        config.SwitchUseFeedFetchOps(false);
        // true for multiple input
        config.SwitchSpecifyInputNames(true);

        config.SwitchIrOptim(true);

        config.EnableMemoryOptim();
        config.DisableGlogInfo();

        this->predictor_ = CreatePredictor(config);
    }

    void FeatureExtracter::Run(cv::Mat &img, std::vector<float> &out_data,
                               std::vector<double> &times) {
        cv::Mat resize_img;
        std::vector<double> time;

        auto preprocess_start = std::chrono::steady_clock::now();
        this->resize_op_.Run(img, resize_img, this->resize_short_,
                             this->resize_size_);

        this->normalize_op_.Run(&resize_img, this->mean_, this->std_, this->scale_);
        std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
        this->permute_op_.Run(&resize_img, input.data());

        auto input_names = this->predictor_->GetInputNames();
        auto input_t = this->predictor_->GetInputHandle(input_names[0]);
        input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
        auto preprocess_end = std::chrono::steady_clock::now();

        auto infer_start = std::chrono::steady_clock::now();
        input_t->CopyFromCpu(input.data());
        this->predictor_->Run();

        auto output_names = this->predictor_->GetOutputNames();
        auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
        std::vector<int> output_shape = output_t->shape();
        int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                      std::multiplies<int>());

        out_data.resize(out_num);
        output_t->CopyToCpu(out_data.data());
        auto infer_end = std::chrono::steady_clock::now();

        auto postprocess_start = std::chrono::steady_clock::now();
        if (this->feature_norm)
            FeatureNorm(out_data);
        auto postprocess_end = std::chrono::steady_clock::now();

        std::chrono::duration<float> preprocess_diff =
                preprocess_end - preprocess_start;
        time.push_back(double(preprocess_diff.count()) * 1000);
        std::chrono::duration<float> inference_diff = infer_end - infer_start;
        double inference_cost_time = double(inference_diff.count() * 1000);
        time.push_back(inference_cost_time);
        // std::chrono::duration<float> postprocess_diff =
        //     postprocess_end - postprocess_start;
        time.push_back(0);

        // std::cout << "result: " << std::endl;
        // std::cout << "\tclass id: " << maxPosition << std::endl;
        // std::cout << std::fixed << std::setprecision(10)
        //           << "\tscore: " << double(out_data[maxPosition]) << std::endl;
        times[0] += time[0];
        times[1] += time[1];
        times[2] += time[2];
    }

    void FeatureExtracter::FeatureNorm(std::vector<float> &featuer) {
        float featuer_sqrt = std::sqrt(std::inner_product(
                featuer.begin(), featuer.end(), featuer.begin(), 0.0f));
        for (int i = 0; i < featuer.size(); ++i)
            featuer[i] /= featuer_sqrt;
    }
} // namespace Feature
