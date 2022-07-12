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

#include <algorithm>
#include <ctime>
#include <include/feature_extractor.h>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

namespace PPShiTu {

// Object Detection Result
struct ObjectResult {
  // Rectangle coordinates of detected object: left, right, top, down
  std::vector<int> rect;
  // Class id of detected object
  int class_id;
  // Confidence of detected object
  float confidence;

  // RecModel result
  std::vector<RESULT> rec_result;
};

void nms(std::vector<ObjectResult> &input_boxes, float nms_threshold,
         bool rec_nms = false);

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                        const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
}

float RectOverlap(const ObjectResult &a, const ObjectResult &b);

inline void
GetMaxScoreIndex(const std::vector<ObjectResult> &det_result,
                 const float threshold,
                 std::vector<std::pair<float, int>> &score_index_vec);

void NMSBoxes(const std::vector<ObjectResult> det_result,
              const float score_threshold, const float nms_threshold,
              std::vector<int> &indices);
} // namespace PPShiTu
