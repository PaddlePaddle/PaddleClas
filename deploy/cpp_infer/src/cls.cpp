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

#include <include/cls.h>

namespace PaddleClas {

void Classifier::LoadModel(const std::string &model_dir) {
  AnalysisConfig config;
  config.SetModel(model_dir + "/model", model_dir + "/params");

  if (this->use_gpu_) {
    config.EnableUseGpu(this->gpu_mem_, this->gpu_id_);
  } else {
    config.DisableGpu();
    if (this->use_mkldnn_) {
      config.EnableMKLDNN();
      // cache 10 different shapes for mkldnn to avoid memory leak
      config.SetMkldnnCacheCapacity(10);
    }
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);
  }

  // false for zero copy tensor
  // true for commom tensor
  config.SwitchUseFeedFetchOps(!this->use_zero_copy_run_);
  // true for multiple input
  config.SwitchSpecifyInputNames(true);

  config.SwitchIrOptim(true);

  config.EnableMemoryOptim();
  config.DisableGlogInfo();

  this->predictor_ = CreatePaddlePredictor(config);
}

void Classifier::Run(cv::Mat &img) {
  cv::Mat srcimg;
  cv::Mat resize_img;
  img.copyTo(srcimg);

  this->resize_op_.Run(img, resize_img, this->resize_short_size_);

  this->crop_op_.Run(resize_img, this->crop_size_);

  this->normalize_op_.Run(&resize_img, this->mean_, this->scale_,
                          this->is_scale_);
  std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
  this->permute_op_.Run(&resize_img, input.data());

  // Inference.
  if (this->use_zero_copy_run_) {
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputTensor(input_names[0]);
    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    input_t->copy_from_cpu(input.data());
    this->predictor_->ZeroCopyRun();
  } else {
    paddle::PaddleTensor input_t;
    input_t.shape = {1, 3, resize_img.rows, resize_img.cols};
    input_t.data =
        paddle::PaddleBuf(input.data(), input.size() * sizeof(float));
    input_t.dtype = PaddleDType::FLOAT32;
    std::vector<paddle::PaddleTensor> outputs;
    this->predictor_->Run({input_t}, &outputs, 1);
  }

  std::vector<float> out_data;
  auto output_names = this->predictor_->GetOutputNames();
  auto output_t = this->predictor_->GetOutputTensor(output_names[0]);
  std::vector<int> output_shape = output_t->shape();
  int out_num = std::accumulate(output_shape.begin(), output_shape.end(), 1,
                                std::multiplies<int>());

  out_data.resize(out_num);
  output_t->copy_to_cpu(out_data.data());

  int maxPosition =
      max_element(out_data.begin(), out_data.end()) - out_data.begin();
  std::cout << "result: " << std::endl;
  std::cout << "\tclass id: " << maxPosition << std::endl;
  std::cout << std::fixed << std::setprecision(10)
            << "\tscore: " << double(out_data[maxPosition]) << std::endl;
}

} // namespace PaddleClas
