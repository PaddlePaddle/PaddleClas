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

#pragma once
#include "paddle_api.h" // NOLINT
#include "json/json.h"
#include <arm_neon.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <sys/time.h>
#include <vector>

using namespace paddle::lite_api; // NOLINT
using namespace std;

namespace PPShiTu {

struct RESULT {
  std::string class_name;
  int class_id;
  float score;
};

class Recognition {

public:
  explicit Recognition(const Json::Value &config_file) {
    MobileConfig config;
    if (config_file["Global"]["rec_model_path"].as<std::string>().empty()) {
      std::cout << "Please set [rec_model_path] in config file" << std::endl;
      exit(-1);
    }
    config.set_model_from_file(
        config_file["Global"]["rec_model_path"].as<std::string>());
    this->predictor = CreatePaddlePredictor<MobileConfig>(config);

    if (config_file["Global"]["rec_label_path"].as<std::string>().empty()) {
      std::cout << "Please set [rec_label_path] in config file" << std::endl;
      exit(-1);
    }
    LoadLabel(config_file["Global"]["rec_label_path"].as<std::string>());
    SetPreProcessParam(config_file["RecPreProcess"]["transform_ops"]);
    if (!config_file["Global"].isMember("return_k")){
      this->topk = config_file["Global"]["return_k"].as<int>();
    }
    printf("rec model create!\n");
  }

  void LoadLabel(std::string path) {
    std::ifstream file;
    std::vector<std::string> label_list;
    file.open(path);
    while (file) {
      std::string line;
      std::getline(file, line);
      std::string::size_type pos = line.find(" ");
      if (pos != std::string::npos) {
        line = line.substr(pos);
      }
      this->label_list.push_back(line);
    }
    file.clear();
    file.close();
  }

  void SetPreProcessParam(const Json::Value &config_file) {
    for (const auto &item : config_file) {
      auto op_name = item["type"].as<std::string>();
      if (op_name == "ResizeImage") {
        this->size = item["size"].as<int>();
      } else if (op_name == "NormalizeImage") {
        this->mean.clear();
        this->std.clear();
        for (auto tmp : item["mean"]) {
          this->mean.emplace_back(tmp.as<float>());
        }
        for (auto tmp : item["std"]) {
          this->std.emplace_back(1 / tmp.as<float>());
        }
        this->scale = item["scale"].as<double>();
      }
    }
  }

  std::vector<RESULT> RunRecModel(const cv::Mat &img, double &cost_time);
  std::vector<RESULT> PostProcess(const float *output_data, int output_size,
                                  cv::Mat &output_image);
  cv::Mat ResizeImage(const cv::Mat &img);
  void NeonMeanScale(const float *din, float *dout, int size);

private:
  std::shared_ptr<PaddlePredictor> predictor;
  std::vector<std::string> label_list;
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
  double scale = 0.00392157;
  float size = 224;
  int topk = 5;
};
} // namespace PPShiTu
