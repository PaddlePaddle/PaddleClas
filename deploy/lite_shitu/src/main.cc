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

#include <algorithm>
#include <cmath>
#include <iostream>
#include <math.h>
#include <numeric>
#include <stdarg.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>

#include "include/config_parser.h"
#include "include/feature_extractor.h"
#include "include/object_detector.h"
#include "include/preprocess_op.h"
#include "include/vector_search.h"
#include "json/json.h"

Json::Value RT_Config;

static std::string DirName(const std::string &filepath) {
  auto pos = filepath.rfind(OS_PATH_SEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

static bool PathExists(const std::string &path) {
  struct stat buffer;
  return (stat(path.c_str(), &buffer) == 0);
}

static void MkDir(const std::string &path) {
  if (PathExists(path))
    return;
  int ret = 0;
  ret = mkdir(path.c_str(), 0755);
  if (ret != 0) {
    std::string path_error(path);
    path_error += " mkdir failed!";
    throw std::runtime_error(path_error);
  }
}

static void MkDirs(const std::string &path) {
  if (path.empty())
    return;
  if (PathExists(path))
    return;

  MkDirs(DirName(path));
  MkDir(path);
}

void DetPredictImage(const std::vector<cv::Mat> &batch_imgs,
                     std::vector<PPShiTu::ObjectResult> &im_result,
                     const int batch_size_det, const int max_det_num,
                     const bool run_benchmark, PPShiTu::ObjectDetector *det) {
  std::vector<double> det_t = {0, 0, 0};
  int steps = ceil(float(batch_imgs.size()) / batch_size_det);
  for (int idx = 0; idx < steps; idx++) {
    int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    // Store all detected result
    std::vector<PPShiTu::ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    bool is_rbox = false;
    if (run_benchmark) {
      det->Predict(batch_imgs, 50, 50, &result, &bbox_num, &det_times);
    } else {
      det->Predict(batch_imgs, 0, 1, &result, &bbox_num, &det_times);
    }

    int item_start_idx = 0;
    for (int i = 0; i < left_image_cnt; i++) {
      cv::Mat im = batch_imgs[i];
      // std::vector<PPShiTu::ObjectResult> im_result;
      int detect_num = 0;
      for (int j = 0; j < min(bbox_num[i], max_det_num); j++) {
        PPShiTu::ObjectResult item = result[item_start_idx + j];
        if (item.class_id == -1) {
          continue;
        }
        detect_num += 1;
        im_result.push_back(item);
      }
      item_start_idx = item_start_idx + bbox_num[i];
    }

    det_t[0] += det_times[0];
    det_t[1] += det_times[1];
    det_t[2] += det_times[2];
  }
}

void PrintResult(std::string &img_path,
                 std::vector<PPShiTu::ObjectResult> &det_result,
                 PPShiTu::VectorSearch &vector_search,
                 PPShiTu::SearchResult &search_result) {
  printf("%s:\n", img_path.c_str());
  for (int i = 0; i < det_result.size(); ++i) {
    int t = i;
    printf("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
           det_result[t].rect[0], det_result[t].rect[1], det_result[t].rect[2],
           det_result[t].rect[3], det_result[t].confidence,
           vector_search.GetLabel(search_result.I[search_result.return_k * t])
               .c_str());
  }
}

int main(int argc, char **argv) {
  std::cout << "Usage: " << argv[0]
            << " [config_path](option) [image_dir](option)\n";
  if (argc < 2) {
    std::cout << "Usage: ./main det_runtime_config.json" << std::endl;
    return -1;
  }
  std::string config_path = argv[1];
  std::string img_dir = "";

  if (argc >= 3) {
    img_dir = argv[2];
  }
  // Parsing command-line
  PPShiTu::load_jsonf(config_path, RT_Config);
  if (RT_Config["Global"]["det_model_path"].as<std::string>().empty()) {
    std::cout << "Please set [det_model_path] in " << config_path << std::endl;
    return -1;
  }

  if (!RT_Config["Global"]["infer_imgs_dir"].as<std::string>().empty() &&
      img_dir.empty()) {
    img_dir = RT_Config["Global"]["infer_imgs_dir"].as<std::string>();
  }
  if (RT_Config["Global"]["infer_imgs"].as<std::string>().empty() &&
      img_dir.empty()) {
    std::cout << "Please set [infer_imgs] in " << config_path
              << " Or use command: <" << argv[0] << " [shitu_config]"
              << " [image_dir]>" << std::endl;
    return -1;
  }
  // Load model and create a object detector
  PPShiTu::ObjectDetector det(
      RT_Config, RT_Config["Global"]["det_model_path"].as<std::string>(),
      RT_Config["Global"]["cpu_num_threads"].as<int>(),
      RT_Config["Global"]["batch_size"].as<int>());
  // create rec model
  PPShiTu::FeatureExtract rec(RT_Config);
  PPShiTu::VectorSearch searcher(RT_Config);
  // Do inference on input image

  std::vector<PPShiTu::ObjectResult> det_result;
  std::vector<cv::Mat> batch_imgs;

  // for vector search
  std::vector<float> feature;
  std::vector<float> features;
  double rec_time;
  if (!RT_Config["Global"]["infer_imgs"].as<std::string>().empty() ||
      !img_dir.empty()) {
    std::vector<std::string> all_img_paths;
    std::vector<cv::String> cv_all_img_paths;
    if (!RT_Config["Global"]["infer_imgs"].as<std::string>().empty()) {
      all_img_paths.push_back(
          RT_Config["Global"]["infer_imgs"].as<std::string>());
      if (RT_Config["Global"]["batch_size"].as<int>() > 1) {
        std::cout << "batch_size_det should be 1, when set `image_file`."
                  << std::endl;
        return -1;
      }
    } else {
      cv::glob(img_dir, cv_all_img_paths);
      for (const auto &img_path : cv_all_img_paths) {
        all_img_paths.push_back(img_path);
      }
    }
    for (int i = 0; i < all_img_paths.size(); ++i) {
      std::string img_path = all_img_paths[i];
      cv::Mat srcimg = cv::imread(img_path, cv::IMREAD_COLOR);
      if (!srcimg.data) {
        std::cerr << "[ERROR] image read failed! image path: " << img_path
                  << "\n";
        exit(-1);
      }
      cv::cvtColor(srcimg, srcimg, cv::COLOR_BGR2RGB);
      batch_imgs.push_back(srcimg);
      DetPredictImage(
          batch_imgs, det_result, RT_Config["Global"]["batch_size"].as<int>(),
          RT_Config["Global"]["max_det_results"].as<int>(), false, &det);

      // add the whole image for recognition to improve recall
//      PPShiTu::ObjectResult result_whole_img = {
//          {0, 0, srcimg.cols, srcimg.rows}, 0, 1.0};
//      det_result.push_back(result_whole_img);

      // get rec result
      PPShiTu::SearchResult search_result;
      for (int j = 0; j < det_result.size(); ++j) {
        int w = det_result[j].rect[2] - det_result[j].rect[0];
        int h = det_result[j].rect[3] - det_result[j].rect[1];
        cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
        cv::Mat crop_img = srcimg(rect);
        rec.RunRecModel(crop_img, rec_time, feature);
        features.insert(features.end(), feature.begin(), feature.end());
      }

      // do vectore search
      search_result = searcher.Search(features.data(), det_result.size());
      PrintResult(img_path, det_result, searcher, search_result);

      batch_imgs.clear();
      det_result.clear();
    }
  }
  return 0;
}
