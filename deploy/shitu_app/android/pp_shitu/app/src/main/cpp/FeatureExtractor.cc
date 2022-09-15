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

#include "FeatureExtractor.h" // NOLINT
#include <utility>            // NOLINT

void FeatureExtract::RunRecModel(const cv::Mat &img, double &cost_time,
                                 std::vector<float> &feature) {
  // Read img
  cv::Mat img_fp;
  ResizeImage(img, img_fp);
  NormalizeImage(&img_fp, this->mean_, this->std_, this->scale_);
  std::vector<float> input(1 * 3 * img_fp.rows * img_fp.cols, 0.0f);
  Permute(&img_fp, input.data());
  auto pre_cost0 = GetCurrentUS();

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(
      std::move(this->predictor_->GetInput(0)));
  input_tensor->Resize({1, 3, this->size, this->size});
  auto *data0 = input_tensor->mutable_data<float>();

  for (int i = 0; i < input.size(); ++i) {
    data0[i] = input[i];
  }
  auto start = std::chrono::system_clock::now();
  // Run predictor
  this->predictor_->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(this->predictor_->GetOutput(0)));
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
}

void FeatureExtract::FeatureNorm(std::vector<float> &feature) {
  float feature_sqrt = std::sqrt(std::inner_product(
      feature.begin(), feature.end(), feature.begin(), 0.0f));
  for (int i = 0; i < feature.size(); ++i) {
    feature[i] /= feature_sqrt;
  }
}

void FeatureExtract::Permute(const cv::Mat *im, float *data) {
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, data + i * rh * rw), i);
  }
}

void FeatureExtract::ResizeImage(const cv::Mat &img, cv::Mat &resize_img) {
  cv::resize(img, resize_img, cv::Size(this->size, this->size));
}

void FeatureExtract::NormalizeImage(cv::Mat *im, const std::vector<float> &mean,
                                    const std::vector<float> &std,
                                    float scale) {
  (*im).convertTo(*im, CV_32FC3, scale);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / std[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / std[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / std[2];
    }
  }
}
