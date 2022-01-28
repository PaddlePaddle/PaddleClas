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

#include "include/recognition.h"

namespace PPShiTu {
std::vector<RESULT> Recognition::RunRecModel(const cv::Mat &img,
                                             double &cost_time) {

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
      std::move(this->predictor->GetOutput(1)));
  auto *output_data = output_tensor->data<float>();
  auto end = std::chrono::system_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  cost_time = double(duration.count()) *
              std::chrono::microseconds::period::num /
              std::chrono::microseconds::period::den;

  int output_size = 1;
  for (auto dim : output_tensor->shape()) {
    output_size *= dim;
  }

  cv::Mat output_image;
  auto results = PostProcess(output_data, output_size, output_image);
  return results;
}

void Recognition::NeonMeanScale(const float *din, float *dout, int size) {

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

cv::Mat Recognition::ResizeImage(const cv::Mat &img) {
  cv::Mat resize_img;
  cv::resize(img, resize_img, cv::Size(this->size, this->size));
  return resize_img;
}
std::vector<RESULT> Recognition::PostProcess(const float *output_data,
                                             int output_size,
                                             cv::Mat &output_image) {

  int max_indices[this->topk];
  double max_scores[this->topk];
  for (int i = 0; i < this->topk; i++) {
    max_indices[i] = 0;
    max_scores[i] = 0;
  }
  for (int i = 0; i < output_size; i++) {
    float score = output_data[i];
    int index = i;
    for (int j = 0; j < this->topk; j++) {
      if (score > max_scores[j]) {
        index += max_indices[j];
        max_indices[j] = index - max_indices[j];
        index -= max_indices[j];
        score += max_scores[j];
        max_scores[j] = score - max_scores[j];
        score -= max_scores[j];
      }
    }
  }

  std::vector<RESULT> results(this->topk);
  for (int i = 0; i < results.size(); i++) {
    results[i].class_name = "Unknown";
    if (max_indices[i] >= 0 && max_indices[i] < this->label_list.size()) {
      results[i].class_name = this->label_list[max_indices[i]];
    }
    results[i].score = max_scores[i];
    results[i].class_id = max_indices[i];
  }
  return results;
}
}
