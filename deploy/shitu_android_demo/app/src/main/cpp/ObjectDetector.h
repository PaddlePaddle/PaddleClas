//   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "Utils.h"                     // NOLINT
#include "paddle_api.h"                // NOLINT
#include <algorithm>                   // NOLINT
#include <functional>                  // NOLINT
#include <iostream>                    // NOLINT
#include <memory>                      // NOLINT
#include <opencv2/core/core.hpp>       // NOLINT
#include <opencv2/highgui/highgui.hpp> // NOLINT
#include <opencv2/imgproc/imgproc.hpp> // NOLINT
#include <stdio.h>                     // NOLINT
#include <string>                      // NOLINT
#include <unordered_map>               // NOLINT
#include <utility>                     // NOLINT
#include <vector>                      // NOLINT

using namespace paddle::lite_api; // NOLINT

struct ObjectPreprocessParam {
  // Normalisze
  std::vector<float> mean;
  std::vector<float> std;
  bool is_scale;
  // resize
  int interp;
  bool keep_ratio;
  std::vector<int> target_size;
  // Pad
  int stride;
  // TopDownEvalAffine
  std::vector<int> trainsize;
};

using PreprocessFunc = std::function<void(cv::Mat *im, ImageBlob *data,
                                          ObjectPreprocessParam item)>;

// PreProcess Function
inline void InitInfo(cv::Mat *im, ImageBlob *data, ObjectPreprocessParam item) {
  data->im_shape_ = {static_cast<float>(im->rows),
                     static_cast<float>(im->cols)};
  data->scale_factor_ = {1., 1.};
  data->in_net_shape_ = {static_cast<float>(im->rows),
                         static_cast<float>(im->cols)};
}

inline void NormalizeImage(cv::Mat *im, ImageBlob *data,
                           ObjectPreprocessParam item) {
  std::vector<float> mean;
  std::vector<float> scale;
  bool is_scale;
  for (auto tmp : item.mean) {
    mean.emplace_back(tmp);
  }
  for (auto tmp : item.std) {
    scale.emplace_back(tmp);
  }
  is_scale = item.is_scale;
  double e = 1.0;
  if (is_scale) {
    e *= 1. / 255.0;
  }
  (*im).convertTo(*im, CV_32FC3, e);
  for (int h = 0; h < im->rows; h++) {
    for (int w = 0; w < im->cols; w++) {
      im->at<cv::Vec3f>(h, w)[0] =
          (im->at<cv::Vec3f>(h, w)[0] - mean[0]) / scale[0];
      im->at<cv::Vec3f>(h, w)[1] =
          (im->at<cv::Vec3f>(h, w)[1] - mean[1]) / scale[1];
      im->at<cv::Vec3f>(h, w)[2] =
          (im->at<cv::Vec3f>(h, w)[2] - mean[2]) / scale[2];
    }
  }
}

inline void Permute(cv::Mat *im, ImageBlob *data, ObjectPreprocessParam item) {
  (*im).convertTo(*im, CV_32FC3);
  int rh = im->rows;
  int rw = im->cols;
  int rc = im->channels();
  (data->im_data_).resize(rc * rh * rw);
  float *base = (data->im_data_).data();
  for (int i = 0; i < rc; ++i) {
    cv::extractChannel(*im, cv::Mat(rh, rw, CV_32FC1, base + i * rh * rw), i);
  }
}

inline void Resize(cv::Mat *im, ImageBlob *data, ObjectPreprocessParam item) {
  std::vector<int> target_size;
  int interp = item.interp;
  bool keep_ratio = item.keep_ratio;
  for (auto tmp : item.target_size) {
    target_size.emplace_back(tmp);
  }
  std::pair<float, float> resize_scale;
  int origin_w = im->cols;
  int origin_h = im->rows;
  if (keep_ratio) {
    int im_size_max = std::max(origin_w, origin_h);
    int im_size_min = std::min(origin_w, origin_h);
    int target_size_max =
        *std::max_element(target_size.begin(), target_size.end());
    int target_size_min =
        *std::min_element(target_size.begin(), target_size.end());
    float scale_min =
        static_cast<float>(target_size_min) / static_cast<float>(im_size_min);
    float scale_max =
        static_cast<float>(target_size_max) / static_cast<float>(im_size_max);
    float scale_ratio = std::min(scale_min, scale_max);
    resize_scale = {scale_ratio, scale_ratio};
  } else {
    resize_scale.first =
        static_cast<float>(target_size[1]) / static_cast<float>(origin_w);
    resize_scale.second =
        static_cast<float>(target_size[0]) / static_cast<float>(origin_h);
  }
  data->im_shape_ = {static_cast<float>(im->cols * resize_scale.first),
                     static_cast<float>(im->rows * resize_scale.second)};
  data->in_net_shape_ = {static_cast<float>(im->cols * resize_scale.first),
                         static_cast<float>(im->rows * resize_scale.second)};
  cv::resize(*im, *im, cv::Size(), resize_scale.first, resize_scale.second,
             interp);
  data->im_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
  data->scale_factor_ = {
      resize_scale.second, resize_scale.first,
  };
}

inline void PadStride(cv::Mat *im, ImageBlob *data,
                      ObjectPreprocessParam item) {
  int stride = item.stride;
  if (stride <= 0) {
    return;
  }
  int rc = im->channels();
  int rh = im->rows;
  int rw = im->cols;
  int nh = (rh / stride) * stride + (rh % stride != 0) * stride;
  int nw = (rw / stride) * stride + (rw % stride != 0) * stride;
  cv::copyMakeBorder(*im, *im, 0, nh - rh, 0, nw - rw, cv::BORDER_CONSTANT,
                     cv::Scalar(0));
  data->in_net_shape_ = {
      static_cast<float>(im->rows), static_cast<float>(im->cols),
  };
}

inline void TopDownEvalAffine(cv::Mat *im, ImageBlob *data,
                              ObjectPreprocessParam item) {
  int interp = 1;
  std::vector<int> trainsize;
  for (auto tmp : item.trainsize) {
    trainsize.emplace_back(tmp);
  }
  cv::resize(*im, *im, cv::Size(trainsize[0], trainsize[1]), 0, 0, interp);
  // todo: Simd::ResizeBilinear();
  data->in_net_shape_ = {
      static_cast<float>(trainsize[1]), static_cast<float>(trainsize[0]),
  };
}

class ObjectDetector {
public: // NOLINT
  explicit ObjectDetector(const std::string &model_dir,
                          std::vector<int> det_input_shape,
                          const int &cpu_threads, std::string cpu_power,
                          const int &batch_size = 1) {
    // global
    fpn_stride_ = std::vector<int>({8, 16, 32, 64});
    // Init preprocess param
    // Normalisze
    pre_param_.mean = std::vector<float>({0.485, 0.456, 0.406});
    pre_param_.std = std::vector<float>({0.229, 0.224, 0.225});
    pre_param_.is_scale = true;
    // resize
    pre_param_.interp = 2;
    pre_param_.keep_ratio = false;
    pre_param_.target_size =
        std::vector<int>({det_input_shape[2], det_input_shape[3]});
    // Pad
    pre_param_.stride = 0;
    // TopDownEvalAffine
    pre_param_.trainsize =
        std::vector<int>({det_input_shape[2], det_input_shape[3]});
    // op
    preprocess_op_func_ = std::vector<std::string>(
        {"DetResize", "DetNormalizeImage", "DetPermute"});
    // init postprocess param
    arch_ = "PicoDet";
    nms_threshold_ = 0.5f;
    score_threshold_ = 0.3f;
    op_map_["InitInfo"] = (PreprocessFunc)InitInfo;
    op_map_["DetNormalizeImage"] = (PreprocessFunc)NormalizeImage;
    op_map_["DetPermute"] = (PreprocessFunc)Permute;
    op_map_["DetResize"] = (PreprocessFunc)Resize;
    op_map_["DetPadStride"] = (PreprocessFunc)PadStride;
    op_map_["DetTopDownEvalAffine"] = (PreprocessFunc)TopDownEvalAffine;
    for (auto op_name : preprocess_op_func_) {
      ops_.emplace_back(op_map_[op_name]);
    }
    LoadModel(model_dir, cpu_threads, cpu_power);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_file, int num_theads,
                 std::string cpu_power);

  // Run predictor
  void Predict(const std::vector<cv::Mat> &imgs, const int warmup = 0,
               const int repeats = 1,
               std::vector<ObjectResult> *result = nullptr,
               std::vector<int> *bbox_num = nullptr,
               std::vector<double> *times = nullptr);

private: // NOLINT
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat &image_mat);

  // Postprocess result
  void Postprocess(const std::vector<cv::Mat> mats,
                   std::vector<ObjectResult> *result, std::vector<int> bbox_num,
                   bool is_rbox);

  std::shared_ptr<PaddlePredictor> predictor_;

  ObjectPreprocessParam pre_param_;
  std::vector<PreprocessFunc> ops_;
  std::vector<std::string> preprocess_op_func_;

  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<int> out_bbox_num_data_;
  float nms_threshold_;
  std::unordered_map<std::string, PreprocessFunc> op_map_;
  std::string arch_;
  float score_threshold_;

  std::vector<int> fpn_stride_;
};
