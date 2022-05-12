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

namespace PPShiTu {
void FeatureExtract::RunRecModel(const cv::Mat &img,
                                 double &cost_time,
                                 std::vector<float> &feature) {
  // Read img
  cv::Mat resize_image = ResizeImage(img);

  cv::Mat img_fp;
  resize_image.convertTo(img_fp, CV_32FC3, scale);

  // Prepare input data from image
  std::unique_ptr<Tensor> input_tensor(std::move(this->predictor->GetInput(0)));
  input_tensor->Resize({1, 3, img_fp.rows, img_fp.cols});
  auto *data0 = input_tensor->mutable_data<float>();

  const float *dimg = reinterpret_cast<const float *>(img_fp.data);
  NeonMeanScale(dimg, data0, img_fp.rows * img_fp.cols);

  auto start = std::chrono::system_clock::now();
  // Run predictor
  this->predictor->Run();

  // Get output and post process
  std::unique_ptr<const Tensor> output_tensor(
      std::move(this->predictor->GetOutput(0)));  //only one output
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  cost_time = double(duration.count()) *
              std::chrono::microseconds::period::num /
              std::chrono::microseconds::period::den;

  //do postprocess
  int output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }
  feature.resize(output_size);
  output_tensor->CopyToCpu(feature.data());

  //postprocess include sqrt or binarize.
  //PostProcess(feature);
  return;
}

// void FeatureExtract::PostProcess(std::vector<float> &feature){
//     float feature_sqrt = std::sqrt(std::inner_product(
//             feature.begin(), feature.end(), feature.begin(), 0.0f));
//     for (int i = 0; i < feature.size(); ++i)
//         feature[i] /= feature_sqrt;
// }

void FeatureExtract::NeonMeanScale(const float *din, float *dout, int size) {

  if (this->mean.size() != 3 || this->std.size() != 3) {
    std::cerr << "[ERROR] mean or scale size must equal to 3\n";
    exit(1);
  }
  float32x4_t vmean0 = vdupq_n_f32(mean[0]);
  float32x4_t vmean1 = vdupq_n_f32(mean[1]);
  float32x4_t vmean2 = vdupq_n_f32(mean[2]);
  float32x4_t vscale0 = vdupq_n_f32(std[0]);
  float32x4_t vscale1 = vdupq_n_f32(std[1]);
  float32x4_t vscale2 = vdupq_n_f32(std[2]);

  float *dout_c0 = dout;
  float *dout_c1 = dout + size;
  float *dout_c2 = dout + size * 2;

  int i = 0;
  for (; i < size - 3; i += 4) {
    float32x4x3_t vin3 = vld3q_f32(din);
    float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
    float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
    float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
    float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
    float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
    float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
    vst1q_f32(dout_c0, vs0);
    vst1q_f32(dout_c1, vs1);
    vst1q_f32(dout_c2, vs2);

    din += 12;
    dout_c0 += 4;
    dout_c1 += 4;
    dout_c2 += 4;
  }
  for (; i < size; i++) {
    *(dout_c0++) = (*(din++) - this->mean[0]) * this->std[0];
    *(dout_c1++) = (*(din++) - this->mean[1]) * this->std[1];
    *(dout_c2++) = (*(din++) - this->mean[2]) * this->std[2];
  }
}

cv::Mat FeatureExtract::ResizeImage(const cv::Mat &img) {
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(this->size, this->size));
  return resize_img;
}
}
