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

#include "include/utils.h"

namespace PPShiTu {

void nms(std::vector<ObjectResult> &input_boxes, float nms_threshold,
         bool rec_nms) {
  if (!rec_nms) {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](ObjectResult a, ObjectResult b) {
                return a.confidence > b.confidence;
              });
  } else {
    std::sort(input_boxes.begin(), input_boxes.end(),
              [](ObjectResult a, ObjectResult b) {
                return a.rec_result[0].score > b.rec_result[0].score;
              });
  }
  std::vector<float> vArea(input_boxes.size());
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    vArea[i] = (input_boxes.at(i).rect[2] - input_boxes.at(i).rect[0] + 1) *
               (input_boxes.at(i).rect[3] - input_boxes.at(i).rect[1] + 1);
  }
  for (int i = 0; i < int(input_boxes.size()); ++i) {
    for (int j = i + 1; j < int(input_boxes.size());) {
      float xx1 = (std::max)(input_boxes[i].rect[0], input_boxes[j].rect[0]);
      float yy1 = (std::max)(input_boxes[i].rect[1], input_boxes[j].rect[1]);
      float xx2 = (std::min)(input_boxes[i].rect[2], input_boxes[j].rect[2]);
      float yy2 = (std::min)(input_boxes[i].rect[3], input_boxes[j].rect[3]);
      float w = (std::max)(float(0), xx2 - xx1 + 1);
      float h = (std::max)(float(0), yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (vArea[i] + vArea[j] - inter);
      if (ovr >= nms_threshold) {
        input_boxes.erase(input_boxes.begin() + j);
        vArea.erase(vArea.begin() + j);
      } else {
        j++;
      }
    }
  }
}

float RectOverlap(const ObjectResult &a, const ObjectResult &b) {
  float Aa = (a.rect[2] - a.rect[0] + 1) * (a.rect[3] - a.rect[1] + 1);
  float Ab = (b.rect[2] - b.rect[0] + 1) * (b.rect[3] - b.rect[1] + 1);

  int iou_w = max(min(a.rect[2], b.rect[2]) - max(a.rect[0], b.rect[0]) + 1, 0);
  int iou_h = max(min(a.rect[3], b.rect[3]) - max(a.rect[1], b.rect[1]) + 1, 0);
  float Aab = iou_w * iou_h;
  return Aab / (Aa + Ab - Aab);
}

inline void
GetMaxScoreIndex(const std::vector<ObjectResult> &det_result,
                 const float threshold,
                 std::vector<std::pair<float, int>> &score_index_vec) {
  // Generate index score pairs.
  for (size_t i = 0; i < det_result.size(); ++i) {
    if (det_result[i].confidence > threshold) {
      score_index_vec.push_back(std::make_pair(det_result[i].confidence, i));
    }
  }

  // Sort the score pair according to the scores in descending order
  std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                   SortScorePairDescend<int>);
}

void NMSBoxes(const std::vector<ObjectResult> det_result,
              const float score_threshold, const float nms_threshold,
              std::vector<int> &indices) {
  int a = 1;
  // Get top_k scores (with corresponding indices).
  std::vector<std::pair<float, int>> score_index_vec;
  GetMaxScoreIndex(det_result, score_threshold, score_index_vec);

  // Do nms
  indices.clear();
  for (size_t i = 0; i < score_index_vec.size(); ++i) {
    const int idx = score_index_vec[i].second;
    bool keep = true;
    for (int k = 0; k < (int)indices.size() && keep; ++k) {
      const int kept_idx = indices[k];
      float overlap = RectOverlap(det_result[idx], det_result[kept_idx]);
      keep = overlap <= nms_threshold;
    }
    if (keep)
      indices.push_back(idx);
  }
}

} // namespace PPShiTu
