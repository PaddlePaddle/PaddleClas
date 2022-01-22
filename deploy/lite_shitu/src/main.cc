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
#include "include/object_detector.h"
#include "include/preprocess_op.h"
#include "include/recognition.h"
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
    std::vector<cv::Mat> batch_imgs;
    int left_image_cnt = batch_imgs.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    /* for (int bs = 0; bs < left_image_cnt; bs++) { */
    /* std::string image_file_path = all_img_paths.at(idx * batch_size_det +
     * bs); */
    /* cv::Mat im = cv::imread(image_file_path, 1); */
    /* batch_imgs.insert(batch_imgs.end(), im); */
    /* } */
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
        /* if (item.rect.size() > 6) { */
        /*   is_rbox = true; */
        /*   printf("class=%d confidence=%.4f rect=[%d %d %d %d %d %d %d %d]\n",
         */
        /*          item.class_id, */
        /*          item.confidence, */
        /*          item.rect[0], */
        /*          item.rect[1], */
        /*          item.rect[2], */
        /*          item.rect[3], */
        /*          item.rect[4], */
        /*          item.rect[5], */
        /*          item.rect[6], */
        /*          item.rect[7]); */
        /* } else { */
        /*   printf("class=%d confidence=%.4f rect=[%d %d %d %d]\n", */
        /*          item.class_id, */
        /*          item.confidence, */
        /*          item.rect[0], */
        /*          item.rect[1], */
        /*          item.rect[2], */
        /*          item.rect[3]); */
        /* } */
      }
      /* std::cout << all_img_paths.at(idx * batch_size_det + i) */
      /* << " The number of detected box: " << detect_num << std::endl; */
      item_start_idx = item_start_idx + bbox_num[i];
    }

    det_t[0] += det_times[0];
    det_t[1] += det_times[1];
    det_t[2] += det_times[2];
  }
}

void PrintResult(const std::string &image_path,
                 std::vector<PPShiTu::ObjectResult> &det_result,
                 std::vector<std::vector<PPShiTu::RESULT>> &rec_results) {
  printf("%s:\n", img_path.c_str());
  for (int i = 0; i < det_result.size(); ++i) {
    printf("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
           det_result[i].rect[0], det_result[i].rect[1], det_result[i].rect[2],
           det_result[t].rect[3], rec_results[i].score,
           rec_results[i].class_name.c_str());
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
  std::string img_path = "";

  if (argc >= 3) {
    img_path = argv[2];
  }
  // Parsing command-line
  PPShiTu::load_jsonf(config_path, RT_Config);
  if (RT_Config["Global"]["det_inference_model_dir"]
          .as<std::string>()
          .empty()) {
    std::cout << "Please set [det_inference_model_dir] in " << config_path
              << std::endl;
    return -1;
  }
  if (RT_Config["Global"]["infer_imgs"].as<std::string>().empty() &&
      img_path.empty()) {
    std::cout << "Please set [infer_imgs] in " << config_path
              << " Or use command: <" << argv[0] << " [shitu_config]"
              << " [image_dir]>" << std::endl;
    return -1;
  }
  if (!img_path.empty()) {
    std::cout << "Use image_dir in command line overide the path in config file"
              << std::endl;
    RT_Config["Global"]["infer_imgs_dir"] = img_path;
    RT_Config["Global"]["infer_imgs"] = "";
  }
  // Load model and create a object detector
  PPShiTu::ObjectDetector det(
      RT_Config,
      RT_Config["Global"]["det_inference_model_dir"].as<std::string>(),
      RT_Config["Global"]["cpu_num_threads"].as<int>(),
      RT_Config["Global"]["batch_size"].as<int>());
  // create rec model
  PPShiTu::Recognition rec(RT_Config);
  // Do inference on input image

  std::vector<PPShiTu::ObjectResult> det_result;
  std::vector<cv::Mat> batch_imgs;
  std::vector<std::vector<PPShiTu::RESULT>> rec_results;
  double rec_time;
  if (!RT_Config["Global"]["infer_imgs"].as<std::string>().empty() ||
      !RT_Config["Global"]["infer_imgs_dir"].as<std::string>().empty()) {
    std::vector<std::string> all_img_paths;
    std::vector<cv::string> cv_all_img_paths;
    if (!RT_Config["Global"]["infer_imgs"].as<std::string>().empty()) {
      all_img_paths.push_back(
          RT_Config["Global"]["infer_imgs"].as<std::string>());
      if (RT_Config["Global"]["batch_size"].as<int>() > 1) {
        std::cout << "batch_size_det should be 1, when set `image_file`."
                  << std::endl;
        return -1;
      }
    } else {
      cv::glob(RT_Config["Global"]["infer_imgs_dir"].as<std::string>(),
               cv_all_img_paths);
      for (const auto &img_path : cv_all_img_paths) {
        all_img_paths.push_back(img_path);
      }
    }
    for (int i = 0; i < all_img_paths.size(); ++i) {
      std::string img_path = img_files_list[idx];
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
      PPShiTu::ObjectResult result_whole_img = {
          {0, 0, srcimg.cols - 1, srcimg.rows - 1}, 0, 1.0};
      det_result.push_back(result_whole_img);

      // get rec result
      for (int j = 0; j < det_result.size(); ++j) {
        int w = det_result[j].rect[2] - det_result[j].rect[0];
        int h = det_result[j].rect[3] - det_result[j].rect[1];
        cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
        cv::Mat crop_img = srcimg(rect);
        std::vector<PPShiTu::RESULT> result =
            rec.RunRecModel(crop_img, rec_time);
        rec_results.push_back(result);
      }
      PrintResult(img_path, det_result, rec_results);

      batch_imgs.clear();
      det_result.clear();
      rec_results.clear();
    }
  }
  return 0;
}
