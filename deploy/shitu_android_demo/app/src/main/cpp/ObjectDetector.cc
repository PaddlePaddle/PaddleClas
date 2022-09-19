//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "ObjectDetector.h" // NOLINT
#include <iomanip>          // NOLINT
#include <sstream>          // NOLINT
#include <unistd.h>
#include <vector> // NOLINT

// PicoDet decode
ObjectResult DisPred2Bbox(const float *dfl_det, int label, float score, int x,
                          int y, int stride, std::vector<float> im_shape,
                          int reg_max) {
  float ct_x = (x + 0.5) * stride;
  float ct_y = (y + 0.5) * stride;
  std::vector<float> dis_pred;
  dis_pred.resize(4);
  for (int i = 0; i < 4; i++) {
    float dis = 0;
    float *dis_after_sm = new float[reg_max + 1];
    activation_function_softmax(dfl_det + i * (reg_max + 1), dis_after_sm,
                                reg_max + 1);
    for (int j = 0; j < reg_max + 1; j++) {
      dis += j * dis_after_sm[j];
    }
    dis *= stride;
    dis_pred[i] = dis;
    delete[] dis_after_sm;
  }
  int xmin = static_cast<int>(std::max(ct_x - dis_pred[0], .0f));
  int ymin = static_cast<int>(std::max(ct_y - dis_pred[1], .0f));
  int xmax = static_cast<int>(std::min(ct_x + dis_pred[2], im_shape[0]));
  int ymax = static_cast<int>(std::min(ct_y + dis_pred[3], im_shape[1]));
  ObjectResult result_item;
  result_item.rect = {xmin, ymin, xmax, ymax};
  result_item.class_id = label;
  result_item.confidence = score;
  return result_item;
}

void PicoDetPostProcess(std::vector<ObjectResult> *results,
                        std::vector<const float *> outs,
                        std::vector<int> fpn_stride,
                        std::vector<float> im_shape,
                        std::vector<float> scale_factor, float score_threshold,
                        float nms_threshold, int num_class, int reg_max) {
  std::vector<std::vector<ObjectResult>> bbox_results;
  bbox_results.resize(num_class);
  int in_h = im_shape[0], in_w = im_shape[1];
  for (int i = 0; i < fpn_stride.size(); ++i) {
    int feature_h = ceil(in_h * 1.f / fpn_stride[i]);
    int feature_w = ceil(in_w * 1.f / fpn_stride[i]);
    for (int idx = 0; idx < feature_h * feature_w; idx++) {
      const float *scores = outs[i] + (idx * num_class);
      int row = idx / feature_w;
      int col = idx % feature_w;
      float score = 0;
      int cur_label = 0;
      for (int label = 0; label < num_class; label++) {
        if (scores[label] > score) {
          score = scores[label];
          cur_label = label;
        }
      }
      if (score > score_threshold) {
        const float *bbox_pred =
            outs[i + fpn_stride.size()] + (idx * 4 * (reg_max + 1));
        bbox_results[cur_label].push_back(
            DisPred2Bbox(bbox_pred, cur_label, score, col, row, fpn_stride[i],
                         im_shape, reg_max));
      }
    }
  }
  for (int i = 0; i < bbox_results.size(); i++) {
    nms(&bbox_results[i], nms_threshold);
    for (auto box : bbox_results[i]) {
      box.rect[0] = box.rect[0] / scale_factor[1];
      box.rect[2] = box.rect[2] / scale_factor[1];
      box.rect[1] = box.rect[1] / scale_factor[0];
      box.rect[3] = box.rect[3] / scale_factor[0];
      results->push_back(box);
    }
  }
}

// ***************************** member Function ***************************
// Load Model and create model predictor
void ObjectDetector::LoadModel(const std::string &model_file, int num_theads,
                               std::string cpu_power) {
  MobileConfig config;
  config.set_threads(num_theads);
  config.set_model_from_file(model_file);
  config.set_power_mode(ParsePowerMode(cpu_power));
  if (access(model_file.c_str(), 0) != 0) {
    LOGD("File not exist!");
  }
  predictor_ = CreatePaddlePredictor<MobileConfig>(config);
}

void ObjectDetector::Preprocess(const cv::Mat &ori_im) {
  // Clone the image : keep the original mat for postprocess
  cv::Mat im = ori_im.clone();
  for (auto &op_process : ops_) {
    op_process(&im, &inputs_, pre_param_);
  }
}

void ObjectDetector::Postprocess(const std::vector<cv::Mat> mats,
                                 std::vector<ObjectResult> *result,
                                 std::vector<int> bbox_num,
                                 bool is_rbox = false) {
  result->clear();
  int start_idx = 0;
  for (int im_id = 0; im_id < mats.size(); im_id++) {
    cv::Mat raw_mat = mats[im_id];
    int rh = 1;
    int rw = 1;
    if (arch_ == "Face") {
      rh = raw_mat.rows;
      rw = raw_mat.cols;
    }
    for (int j = start_idx; j < start_idx + bbox_num[im_id]; j++) {
      if (is_rbox) {
        // Class id
        int class_id = static_cast<int>(round(output_data_[0 + j * 10]));
        // Confidence score
        float score = output_data_[1 + j * 10];
        int x1 = (output_data_[2 + j * 10] * rw);
        int y1 = (output_data_[3 + j * 10] * rh);
        int x2 = (output_data_[4 + j * 10] * rw);
        int y2 = (output_data_[5 + j * 10] * rh);
        int x3 = (output_data_[6 + j * 10] * rw);
        int y3 = (output_data_[7 + j * 10] * rh);
        int x4 = (output_data_[8 + j * 10] * rw);
        int y4 = (output_data_[9 + j * 10] * rh);
        ObjectResult result_item;
        result_item.rect = {x1, y1, x2, y2, x3, y3, x4, y4};
        result_item.class_id = class_id;
        result_item.confidence = score;
        result->push_back(result_item);
      } else {
        // Class id
        int class_id = static_cast<int>(round(output_data_[0 + j * 6]));
        // Confidence score
        float score = output_data_[1 + j * 6];
        int xmin = (output_data_[2 + j * 6] * rw);
        int ymin = (output_data_[3 + j * 6] * rh);
        int xmax = (output_data_[4 + j * 6] * rw);
        int ymax = (output_data_[5 + j * 6] * rh);
        int wd = xmax - xmin;
        int hd = ymax - ymin;

        ObjectResult result_item;
        result_item.rect = {xmin, ymin, xmax, ymax};
        result_item.class_id = class_id;
        result_item.confidence = score;
        result->push_back(result_item);
      }
    }
    start_idx += bbox_num[im_id];
  }
}

void ObjectDetector::Predict(const std::vector<cv::Mat> &imgs, const int warmup,
                             const int repeats,
                             std::vector<ObjectResult> *result,
                             std::vector<int> *bbox_num,
                             std::vector<double> *times) {
  auto preprocess_start = std::chrono::steady_clock::now();
  int batch_size = imgs.size();

  // in_data_batch
  std::vector<float> in_data_all;
  std::vector<float> im_shape_all(batch_size * 2);
  std::vector<float> scale_factor_all(batch_size * 2);
  // Preprocess image
  for (int bs_idx = 0; bs_idx < batch_size; bs_idx++) {
    cv::Mat im = imgs.at(bs_idx);
    Preprocess(im);
    im_shape_all[bs_idx * 2] = inputs_.im_shape_[0];
    im_shape_all[bs_idx * 2 + 1] = inputs_.im_shape_[1];

    scale_factor_all[bs_idx * 2] = inputs_.scale_factor_[0];
    scale_factor_all[bs_idx * 2 + 1] = inputs_.scale_factor_[1];

    // TODO: reduce cost time
    in_data_all.insert(in_data_all.end(), inputs_.im_data_.begin(),
                       inputs_.im_data_.end());
  }
  auto preprocess_end = std::chrono::steady_clock::now();
  std::vector<const float *> output_data_list_;
  // Prepare input tensor

  auto input_names = predictor_->GetInputNames();
  for (const auto &tensor_name : input_names) {
    auto in_tensor = predictor_->GetInputByName(tensor_name);
    if (tensor_name == "image") {
      int rh = inputs_.in_net_shape_[0];
      int rw = inputs_.in_net_shape_[1];
      in_tensor->Resize({batch_size, 3, rh, rw});
      auto *inptr = in_tensor->mutable_data<float>();
      std::copy_n(in_data_all.data(), in_data_all.size(), inptr);
    } else if (tensor_name == "im_shape") {
      in_tensor->Resize({batch_size, 2});
      auto *inptr = in_tensor->mutable_data<float>();
      std::copy_n(im_shape_all.data(), im_shape_all.size(), inptr);
    } else if (tensor_name == "scale_factor") {
      in_tensor->Resize({batch_size, 2});
      auto *inptr = in_tensor->mutable_data<float>();
      std::copy_n(scale_factor_all.data(), scale_factor_all.size(), inptr);
    }
  }

  // Run predictor
  // warmup
  for (int i = 0; i < warmup; i++) {
    predictor_->Run();
    // Get output tensor
    auto output_names = predictor_->GetOutputNames();
    if (arch_ == "PicoDet") {
      for (int j = 0; j < output_names.size(); j++) {
        auto output_tensor = predictor_->GetTensor(output_names[j]);
        const float *outptr = output_tensor->data<float>();
        std::vector<int64_t> output_shape = output_tensor->shape();
        output_data_list_.push_back(outptr);
      }
    } else {
      auto out_tensor = predictor_->GetTensor(output_names[0]);
      auto out_bbox_num = predictor_->GetTensor(output_names[1]);
    }
  }

  bool is_rbox = false;
  auto inference_start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeats; i++) {
    predictor_->Run();
  }
  auto inference_end = std::chrono::steady_clock::now();
  auto postprocess_start = std::chrono::steady_clock::now();

  // Get output tensor
  output_data_list_.clear();
  int num_class = 1;
  int reg_max = 7;
  auto output_names = predictor_->GetOutputNames();
  // TODO: Unified model output.
  if (arch_ == "PicoDet") {
    for (int i = 0; i < output_names.size(); i++) {
      auto output_tensor = predictor_->GetTensor(output_names[i]);
      const float *outptr = output_tensor->data<float>();
      std::vector<int64_t> output_shape = output_tensor->shape();
      if (i == 0) {
        num_class = output_shape[2];
      }
      if (i == fpn_stride_.size()) {
        reg_max = output_shape[2] / 4 - 1;
      }
      output_data_list_.push_back(outptr);
    }
  } else {
    auto output_tensor = predictor_->GetTensor(output_names[0]);
    auto output_shape = output_tensor->shape();
    auto out_bbox_num = predictor_->GetTensor(output_names[1]);
    auto out_bbox_num_shape = out_bbox_num->shape();
    // Calculate output length
    int output_size = 1;
    for (int j = 0; j < output_shape.size(); ++j) {
      output_size *= output_shape[j];
    }
    is_rbox = output_shape[output_shape.size() - 1] % 10 == 0;

    if (output_size < 6) {
      std::cerr << "[WARNING] No object detected." << std::endl;
    }
    output_data_.resize(output_size);
    std::copy_n(output_tensor->mutable_data<float>(), output_size,
                output_data_.data());

    int out_bbox_num_size = 1;
    for (int j = 0; j < out_bbox_num_shape.size(); ++j) {
      out_bbox_num_size *= out_bbox_num_shape[j];
    }
    out_bbox_num_data_.resize(out_bbox_num_size);
    std::copy_n(out_bbox_num->mutable_data<int>(), out_bbox_num_size,
                out_bbox_num_data_.data());
  }
  // Postprocessing result

  result->clear();
  if (arch_ == "PicoDet") {
    PicoDetPostProcess(result, output_data_list_, fpn_stride_,
                       inputs_.im_shape_, inputs_.scale_factor_,
                       score_threshold_, nms_threshold_, num_class, reg_max);
    bbox_num->push_back(result->size());
  } else {
    Postprocess(imgs, result, out_bbox_num_data_, is_rbox);
    bbox_num->clear();
    for (int k = 0; k < out_bbox_num_data_.size(); k++) {
      int tmp = out_bbox_num_data_[k];
      bbox_num->push_back(tmp);
    }
  }
  auto postprocess_end = std::chrono::steady_clock::now();

  std::chrono::duration<float> preprocess_diff =
      preprocess_end - preprocess_start;
  times->push_back(double(preprocess_diff.count() * 1000));
  std::chrono::duration<float> inference_diff = inference_end - inference_start;
  times->push_back(double(inference_diff.count() / repeats * 1000));
  std::chrono::duration<float> postprocess_diff =
      postprocess_end - postprocess_start;
  times->push_back(double(postprocess_diff.count() * 1000));
}
