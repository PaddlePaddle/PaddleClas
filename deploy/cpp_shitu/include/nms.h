// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

// This code is adpated from opencv(https://github.com/opencv/opencv)

#include <algorithm>
#include <include/object_detector.h>

template<typename T>
static inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                        const std::pair<float, T> &pair2) {
    return pair1.first > pair2.first;
}

float RectOverlap(const Detection::ObjectResult &a,
                  const Detection::ObjectResult &b) {
    float Aa = (a.rect[2] - a.rect[0] + 1) * (a.rect[3] - a.rect[1] + 1);
    float Ab = (b.rect[2] - b.rect[0] + 1) * (b.rect[3] - b.rect[1] + 1);

    int iou_w = max(min(a.rect[2], b.rect[2]) - max(a.rect[0], b.rect[0]) + 1, 0);
    int iou_h = max(min(a.rect[3], b.rect[3]) - max(a.rect[1], b.rect[1]) + 1, 0);
    float Aab = iou_w * iou_h;
    return Aab / (Aa + Ab - Aab);
}

// Get max scores with corresponding indices.
//    scores: a set of scores.
//    threshold: only consider scores higher than the threshold.
//    top_k: if -1, keep all; otherwise, keep at most top_k.
//    score_index_vec: store the sorted (score, index) pair.
inline void
GetMaxScoreIndex(const std::vector <Detection::ObjectResult> &det_result,
                 const float threshold,
                 std::vector <std::pair<float, int>> &score_index_vec) {
    // Generate index score pairs.
    for (size_t i = 0; i < det_result.size(); ++i) {
        if (det_result[i].confidence > threshold) {
            score_index_vec.push_back(std::make_pair(det_result[i].confidence, i));
        }
    }

    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec.begin(), score_index_vec.end(),
                     SortScorePairDescend<int>);

    // // Keep top_k scores if needed.
    // if (top_k > 0 && top_k < (int)score_index_vec.size())
    // {
    //     score_index_vec.resize(top_k);
    // }
}

void NMSBoxes(const std::vector <Detection::ObjectResult> det_result,
              const float score_threshold, const float nms_threshold,
              std::vector<int> &indices) {
    int a = 1;
    // Get top_k scores (with corresponding indices).
    std::vector <std::pair<float, int>> score_index_vec;
    GetMaxScoreIndex(det_result, score_threshold, score_index_vec);

    // Do nms
    indices.clear();
    for (size_t i = 0; i < score_index_vec.size(); ++i) {
        const int idx = score_index_vec[i].second;
        bool keep = true;
        for (int k = 0; k < (int) indices.size() && keep; ++k) {
            const int kept_idx = indices[k];
            float overlap = RectOverlap(det_result[idx], det_result[kept_idx]);
            keep = overlap <= nms_threshold;
        }
        if (keep)
            indices.push_back(idx);
    }
}
