//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "paddle_inference_api.h" // NOLINT

#include "include/preprocess_op_det.h"
#include "include/yaml_config.h"

using namespace paddle_infer;

namespace Detection {
// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
};

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

// Visualiztion Detection Result
cv::Mat VisualizeResult(const cv::Mat &img,
                        const std::vector<ObjectResult> &results,
                        const std::vector<std::string> &lables,
                        const std::vector<int> &colormap, const bool is_rbox);

class ObjectDetector {
public:
  explicit ObjectDetector(const YAML::Node &config_file) {
    this->use_gpu_ = config_file["Global"]["use_gpu"].as<bool>();
    if (config_file["Global"]["gpu_id"].IsDefined())
      this->gpu_id_ = config_file["Global"]["gpu_id"].as<int>();
    this->gpu_mem_ = config_file["Global"]["gpu_mem"].as<int>();
    this->cpu_math_library_num_threads_ =
        config_file["Global"]["cpu_num_threads"].as<int>();
    this->use_mkldnn_ = config_file["Global"]["enable_mkldnn"].as<bool>();
    this->use_tensorrt_ = config_file["Global"]["use_tensorrt"].as<bool>();
    this->use_fp16_ = config_file["Global"]["use_fp16"].as<bool>();
    this->model_dir_ =
        config_file["Global"]["det_inference_model_dir"].as<std::string>();
    this->threshold_ = config_file["Global"]["threshold"].as<float>();
    this->max_det_results_ = config_file["Global"]["max_det_results"].as<int>();
    this->image_shape_ =
        config_file["Global"]["image_shape"].as<std::vector<int>>();
    this->label_list_ =
        config_file["Global"]["label_list"].as<std::vector<std::string>>();
    this->ir_optim_ = config_file["Global"]["ir_optim"].as<bool>();
    this->batch_size_ = config_file["Global"]["batch_size"].as<int>();

    preprocessor_.Init(config_file["DetPreProcess"]["transform_ops"]);
    LoadModel(model_dir_, batch_size_, run_mode);
  }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir, const int batch_size = 1,
                 const std::string &run_mode = "fluid");

  // Run predictor
  void Predict(const std::vector<cv::Mat> imgs, const int warmup = 0,
               const int repeats = 1,
               std::vector<ObjectResult> *result = nullptr,
               std::vector<int> *bbox_num = nullptr,
               std::vector<double> *times = nullptr);

  const std::vector<std::string> &GetLabelList() const {
    return this->label_list_;
  }

  const float &GetThreshold() const { return this->threshold_; }

private:
  bool use_gpu_ = true;
  int gpu_id_ = 0;
  int gpu_mem_ = 800;
  int cpu_math_library_num_threads_ = 6;
  std::string run_mode = "fluid";
  bool use_mkldnn_ = false;
  bool use_tensorrt_ = false;
  bool batch_size_ = 1;
  bool use_fp16_ = false;
  std::string model_dir_;
  float threshold_ = 0.5;
  float max_det_results_ = 5;
  std::vector<int> image_shape_ = {3, 640, 640};
  std::vector<std::string> label_list_;
  bool ir_optim_ = true;
  bool det_permute_ = true;
  bool det_postprocess_ = true;
  int min_subgraph_size_ = 30;
  bool use_dynamic_shape_ = false;
  int trt_min_shape_ = 1;
  int trt_max_shape_ = 1280;
  int trt_opt_shape_ = 640;
  bool trt_calib_mode_ = false;

  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat &image_mat);

  // Postprocess result
  void Postprocess(const std::vector<cv::Mat> mats,
                   std::vector<ObjectResult> *result, std::vector<int> bbox_num,
                   bool is_rbox);

  std::shared_ptr<Predictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<int> out_bbox_num_data_;
};

} // namespace Detection
