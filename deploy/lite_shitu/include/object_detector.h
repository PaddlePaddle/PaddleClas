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

#include <ctime>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <stdlib.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "json/json.h"

#include "paddle_api.h"  // NOLINT

#include "include/config_parser.h"
#include "include/preprocess_op.h"
#include "include/utils.h"
#include "include/picodet_postprocess.h"

using namespace paddle::lite_api;  // NOLINT

namespace PPShiTu {

// Generate visualization colormap for each class
std::vector<int> GenerateColorMap(int num_class);

// Visualiztion Detection Result
cv::Mat VisualizeResult(const cv::Mat& img,
                        const std::vector<PPShiTu::ObjectResult>& results,
                        const std::vector<std::string>& lables,
                        const std::vector<int>& colormap,
                        const bool is_rbox);

class ObjectDetector {
 public:
  explicit ObjectDetector(const Json::Value& config,
		  	  const std::string& model_dir,
                          int cpu_threads = 1,
                          const int batch_size = 1) {
    config_.load_config(config);
    printf("config created\n");
    preprocessor_.Init(config_.preprocess_info_);
    printf("before object detector\n");
    if(config["Global"]["det_model_path"].as<std::string>().empty()){
	std::cout << "Please set [det_model_path] in config file" << std::endl;
	exit(-1);
    }
    LoadModel(config["Global"]["det_model_path"].as<std::string>(), cpu_threads);
    printf("create object detector\n"); }

  // Load Paddle inference model
  void LoadModel(std::string model_file, int num_theads);

  // Run predictor
  void Predict(const std::vector<cv::Mat>& imgs,
               const int warmup = 0,
               const int repeats = 1,
               std::vector<PPShiTu::ObjectResult>* result = nullptr,
               std::vector<int>* bbox_num = nullptr,
               std::vector<double>* times = nullptr);

  // Get Model Label list
  const std::vector<std::string>& GetLabelList() const {
    return config_.label_list_;
  }

 private:
  // Preprocess image and copy data to input buffer
  void Preprocess(const cv::Mat& image_mat);
  // Postprocess result
  void Postprocess(const std::vector<cv::Mat> mats,
                   std::vector<PPShiTu::ObjectResult>* result,
                   std::vector<int> bbox_num,
                   bool is_rbox);

  std::shared_ptr<PaddlePredictor> predictor_;
  Preprocessor preprocessor_;
  ImageBlob inputs_;
  std::vector<float> output_data_;
  std::vector<int> out_bbox_num_data_;
  float threshold_;
  ConfigPaser config_;

};

}  // namespace PPShiTu
