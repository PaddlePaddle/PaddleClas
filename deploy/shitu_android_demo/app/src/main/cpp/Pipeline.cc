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

#include "Pipeline.h"
#include "FeatureExtractor.h"
#include "ObjectDetector.h"
#include "VectorSearch.h"
#include <algorithm>
#include <functional>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

void PrintResult(std::vector<ObjectResult> &det_result,
                 const std::shared_ptr<VectorSearch> &vector_search,
                 SearchResult &search_result) {
  for (int i = 0; i < std::min((int)det_result.size(), 1); ++i) {
    int t = i;
    LOGD("\tresult%d: bbox[%d, %d, %d, %d], score: %f, label: %s\n", i,
         det_result[t].rect[0], det_result[t].rect[1], det_result[t].rect[2],
         det_result[t].rect[3], det_result[t].confidence,
         vector_search->GetLabel(search_result.I[search_result.return_k * t])
             .c_str());
  }
}

void VisualResult(cv::Mat &img, std::vector<ObjectResult> &results) { // NOLINT
  for (int i = 0; i < 1; i++) {
    int w = results[i].rect[2] - results[i].rect[0];
    int h = results[i].rect[3] - results[i].rect[1];
    cv::Rect roi = cv::Rect(results[i].rect[0], results[i].rect[1], w, h);
    cv::rectangle(img, roi, cv::Scalar(41, 50, 255), 3);
  }
}

PipeLine::PipeLine(std::string det_model_path, std::string rec_model_path,
                   std::string label_path, std::string index_path,
                   std::vector<int> det_input_shape,
                   std::vector<int> rec_input_shape, int cpu_num_threads,
                   int warm_up, int repeats, int topk, bool add_gallery,
                   std::string cpu_power) {
  det_model_path_ = det_model_path;
  rec_model_path_ = rec_model_path;
  label_path_ = label_path;
  index_path_ = index_path;
  det_input_shape_ = det_input_shape;
  rec_input_shape_ = rec_input_shape;
  cpu_num_threads_ = cpu_num_threads;
  add_gallery_flag = add_gallery;

  max_det_num_ = topk;
  cpu_pow_ = cpu_power;
  det_model_path_ =
      det_model_path_ + "/mainbody_PPLCNet_x2_5_640_quant_v1.0_lite.nb";
  rec_model_path_ =
      rec_model_path_ + "/general_PPLCNetV2_base_quant_v1.0_lite.nb";

  // create object detector
  det_ = std::make_shared<ObjectDetector>(det_model_path_, det_input_shape_,
                                          cpu_num_threads_, cpu_pow_);

  // create rec model
  rec_ = std::make_shared<FeatureExtract>(rec_model_path_, rec_input_shape_,
                                          cpu_num_threads_, cpu_pow_);
  // create vector search
  searcher_ = std::make_shared<VectorSearch>(label_path_, index_path_, 5, 0.5);
}

std::string PipeLine::run(std::vector<cv::Mat> &batch_imgs,      // NOLINT
                          std::vector<ObjectResult> &det_result, // NOLINT
                          int batch_size, const std::string &label_name) {
  std::fill(times_.begin(), times_.end(), 0);

  //    if (!this->add_gallery_flag)
  //    {
  DetPredictImage(batch_imgs, &det_result, batch_size, det_,
                  max_det_num_); // det_result获取[l,d,r,u]
                                 //    }
  // add the whole image for recognition to improve recall

  ObjectResult result_whole_img = {
      {0, 0, batch_imgs[0].cols, batch_imgs[0].rows}, 0, 1.0};
  det_result.push_back(result_whole_img); // 加入整图的坐标，提升召回率
  // get rec result
  for (int j = 0; j < det_result.size(); ++j) {
    double rec_time = 0.0; // .rect:vector = {l, d, r, u}
    vector<float> feature;
    int w = det_result[j].rect[2] - det_result[j].rect[0];
    int h = det_result[j].rect[3] - det_result[j].rect[1];
    cv::Rect rect(det_result[j].rect[0], det_result[j].rect[1], w, h);
    cv::Mat crop_img = batch_imgs[0](rect);
    rec_->RunRecModel(crop_img, rec_time, feature);
    if (this->add_gallery_flag) {
      this->searcher_->AddFeature(feature.data(), label_name);
    } else {
      features.insert(features.end(), feature.begin(),
                      feature.end()); //每次插入一个512的向量
    }
  }
  if (this->add_gallery_flag) {
    VisualResult(batch_imgs[0], det_result);
    det_result.clear();
    features.clear();
    indices.clear();
    std::string res = std::to_string(times_[1] + times_[4]) + "\n";
    return res;
  }
  // do vectore search
  SearchResult search_result = searcher_->Search(
      features.data(),
      det_result
          .size()); // 一次搜索多个向量(展平在features里)，共det_result.size()个
                    //    for (int j = 0; j < det_result.size(); ++j)
  for (int j = 0; j < 1; ++j) // 对于每个检测框，只把
  {
    det_result[j].confidence =
        search_result.return_k * j < search_result.D.size()
            ? search_result.D[search_result.return_k * j]
            : 0.0f;
    for (int k = 0; k < this->max_index_num_; ++k) {
      std::size_t tidx =
          min<std::size_t>((std::size_t)(search_result.return_k * j + k),
                           search_result.D.size() - 1);

      std::string _class_name = searcher_->GetLabel(search_result.I[tidx]);
      int _index = (int)(search_result.I[tidx]);
      float _dist = search_result.D[tidx];
      if (_dist > 1e5 || _dist < -1e5) {
        _dist = 0.0;
      }

      det_result[j].rec_result.push_back({_class_name, _index, _dist});
    }
  }
  //    sort(det_result.begin(), det_result.end(), [](const ObjectResult &a,
  //    const ObjectResult &b){
  //        if (a.rec_result.empty() and b.rec_result.empty())
  //        {
  //            return 0;
  //        }
  //        else if (a.rec_result.empty() and !b.rec_result.empty())
  //        {
  //            return 0;
  //        }
  //        else if (!a.rec_result.empty() and b.rec_result.empty())
  //        {
  //            return 1;
  //        }
  //        else
  //        {
  //            return (int)(a.rec_result[0].score > b.rec_result[0].score);
  //        }
  //    });
  NMSBoxes(det_result, searcher_->GetThreshold(), this->rec_nms_thresold_,
           indices);
  VisualResult(batch_imgs[0], det_result);
  LOGD("================== result summary =========================");
  PrintResult(det_result, searcher_, search_result);

  // results
  std::string res;
  res += std::to_string(times_[1] + times_[4]) + "\n";
  for (int i = 0; i < 1; i++) {
    res.append(det_result[i].rec_result[0].class_name + ", " +
               std::to_string((int)(det_result[i].rec_result[0].score * 1000) *
                              1.0 / 1000) +
               "\n");
  }
  det_result.clear();
  features.clear();
  indices.clear();
  return res;
}

void PipeLine::DetPredictImage(const std::vector<cv::Mat> batch_imgs,
                               std::vector<ObjectResult> *im_result,
                               const int batch_size_det,
                               std::shared_ptr<ObjectDetector> det,
                               const int max_det_num) {
  int steps = ceil(float(batch_imgs.size()) / batch_size_det);
  for (int idx = 0; idx < steps; idx++) {
    int left_image_cnt = (int)batch_imgs.size() - idx * batch_size_det;
    if (left_image_cnt > batch_size_det) {
      left_image_cnt = batch_size_det;
    }
    // Store all detected result
    std::vector<ObjectResult> result;
    std::vector<int> bbox_num;
    std::vector<double> det_times;

    //        bool is_rbox = false;
    det->Predict(batch_imgs, 0, 1, &result, &bbox_num, &det_times);
    int item_start_idx = 0;
    for (int i = 0; i < left_image_cnt; i++) {
      cv::Mat im = batch_imgs[i];
      int detect_num = 0;
      for (int j = 0; j < min(bbox_num[i], max_det_num); j++) {
        ObjectResult item = result[item_start_idx + j];
        if (item.class_id == -1) {
          continue;
        }
        detect_num += 1;
        im_result->push_back(item);
      }
      item_start_idx = item_start_idx + bbox_num[i];
    }
    times_[0] += det_times[0];
    times_[1] += det_times[1];
    times_[2] += det_times[2];
  }
}

template <typename T>
static inline bool SortScorePairDescend(const std::pair<float, T> &pair1,
                                        const std::pair<float, T> &pair2) {
  return pair1.first > pair2.first;
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

float RectOverlap(const ObjectResult &a, const ObjectResult &b) {
  float Aa = (a.rect[2] - a.rect[0] + 1) * (a.rect[3] - a.rect[1] + 1);
  float Ab = (b.rect[2] - b.rect[0] + 1) * (b.rect[3] - b.rect[1] + 1);

  int iou_w = max(min(a.rect[2], b.rect[2]) - max(a.rect[0], b.rect[0]) + 1, 0);
  int iou_h = max(min(a.rect[3], b.rect[3]) - max(a.rect[1], b.rect[1]) + 1, 0);
  float Aab = iou_w * iou_h;
  return Aab / (Aa + Ab - Aab);
}

void PipeLine::NMSBoxes(const std::vector<ObjectResult> &det_result,
                        const float score_threshold, const float nms_threshold,
                        std::vector<int> &indices) {
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

void PipeLine::set_add_gallery(const bool &flag) {
  this->add_gallery_flag = flag;
}

void PipeLine::ClearFeature() { this->searcher_->ClearFeature(); }

void PipeLine::SaveIndex(const string &save_file_name) {
  this->searcher_->SaveIndex(save_file_name);
}

bool PipeLine::LoadIndex(const string &save_file_name) {
  return this->searcher_->LoadFromSaveFileName(save_file_name);
}

string PipeLine::GetLabelList() {
  std::vector<std::string> class_name_list = this->searcher_->GetLabelList();
  string ret;
  ret += "共";
  ret += std::to_string(class_name_list.size());
  ret += "类";
  ret += "\n";
  ret += "====================\n";
  for (const auto &str : class_name_list) {
    ret += str;
    ret += "\n";
  }
  return ret;
}
