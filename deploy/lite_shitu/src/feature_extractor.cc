// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "include/feature_extractor.h"
#include <cmath>
#include <numeric>

namespace PPShiTu {
void FeatureExtract::RunRecModel(const cv::Mat &img, double &cost_time,
                                 std::vector<float> &feature) {
  // Read img
  cv::Mat img_fp;
  this->resize_op_.Run_feature(img, img_fp, this->size, this->size);
  this->normalize_op_.Run_feature(&img_fp, this->mean, this->std, this->scale);
  std::vector<float> input(1 * 3 * img_fp.rows * img_fp.cols, 0.0f);
  this->permute_op_.Run_feature(&img_fp, input.data());

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(this->predictor->GetInput(0)));
  input_tensor->Resize({1, 3, this->size, this->size});
  auto *data0 = input_tensor->mutable_data<float>();

  // const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  // NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols);
  for (int i = 0; i < input.size(); ++i) {
    data0[i] = input[i];
  }

  auto start = std::chrono::system_clock::now();
  // Run predictor
  this->predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(this->predictor->GetOutput(0))); // only one output
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  cost_time = double(duration.count()) *
              std::chrono::microseconds::period::num /
              std::chrono::microseconds::period::den;

  // do postprocess
  int output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  feature.resize(output_size);
  output_tensor->CopyToCpu(feature.data());

  // postprocess include sqrt or binarize.
  FeatureNorm(feature);
  return;
}

void FeatureExtract::FeatureNorm(std::vector<float> &feature) {
  float feature_sqrt = std::sqrt(std::inner_product(
      feature.begin(), feature.end(), feature.begin(), 0.0f));
  for (int i = 0; i < feature.size(); ++i)
    feature[i] /= feature_sqrt;
}
} // namespace PPShiTu
