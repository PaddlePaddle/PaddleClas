// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle_api.h"  // NOLINT
#include <android/log.h> // NOLINT
#include <fstream>       // NOLINT
#include <string>        // NOLINT
#include <vector>        // NOLINT

#define TAG "JNI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)
#define LOGF(...) __android_log_print(ANDROID_LOG_FATAL, TAG, __VA_ARGS__)

#define MIN(a, b) (((a) < (b)) ? (a) : (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

int64_t ShapeProduction(const std::vector<int64_t> &shape);

template <typename T>
bool ReadFile(const std::string &path, std::vector<T> *data) {
  std::ifstream file(path, std::ifstream::binary);
  if (file) {
    file.seekg(0, file.end);
    int size = file.tellg();
    LOGD("file size=%lld\n", size);
    data->resize(size / sizeof(T));
    file.seekg(0, file.beg);
    file.read(reinterpret_cast<char *>(data->data()), size);
    file.close();
    return true;
  } else {
    LOGE("Can't read file from %s\n", path.c_str());
  }
  return false;
}

template <typename T>
bool WriteFile(const std::string &path, const std::vector<T> &data) {
  std::ofstream file{path, std::ios::binary};
  if (!file.is_open()) {
    LOGE("Can't write file to %s\n", path.c_str());
    return false;
  }
  file.write(reinterpret_cast<const char *>(data.data()),
             data.size() * sizeof(T));
  file.close();
  return true;
}

inline int64_t GetCurrentTime() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1000000LL * (int64_t)time.tv_sec + (int64_t)time.tv_usec;
}

inline double GetElapsedTime(int64_t time) {
  return (GetCurrentTime() - time) / 1000.0f;
}

inline paddle::lite_api::PowerMode ParsePowerMode(std::string mode) {
  if (mode == "LITE_POWER_HIGH") {
    return paddle::lite_api::LITE_POWER_HIGH;
  } else if (mode == "LITE_POWER_LOW") {
    return paddle::lite_api::LITE_POWER_LOW;
  } else if (mode == "LITE_POWER_FULL") {
    return paddle::lite_api::LITE_POWER_FULL;
  } else if (mode == "LITE_POWER_RAND_HIGH") {
    return paddle::lite_api::LITE_POWER_RAND_HIGH;
  } else if (mode == "LITE_POWER_RAND_LOW") {
    return paddle::lite_api::LITE_POWER_RAND_LOW;
  }
  return paddle::lite_api::LITE_POWER_NO_BIND;
}

void NHWC3ToNC3HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height);

void NHWC1ToNC1HW(const float *src, float *dst, const float *mean,
                  const float *std, int width, int height);

// Recognise Result
struct RectResult {
  std::string class_name;
  int class_id;
  float score;
};

// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;
  // RecModel result
  std::vector<RectResult> rec_result;
};

// Object for storing all preprocessed data
class ImageBlob {
public: // NOLINT
  // image width and height
  std::vector<float> im_shape_;
  // Buffer for image data after preprocessing
  std::vector<float> im_data_;
  // in net data shape(after pad)
  std::vector<float> in_net_shape_;
  // Evaluation image width and height
  // std::vector<float>  eval_im_size_f_;
  // Scale factor for image size to origin image size
  std::vector<float> scale_factor_;
};

void nms(std::vector<ObjectResult> *input_boxes, float nms_threshold,
         bool rec_nms = false);

void neon_mean_scale(const float *din, float *dout, int size, float *mean,
                     float *scale);

void activation_function_softmax(const float *src, float *dst, int length);

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}
