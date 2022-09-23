// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "core/general-server/op/general_picodet_op.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <sstream>

#include "core/predictor/framework/infer.h"
#include "core/predictor/framework/memory.h"
#include "core/predictor/framework/resource.h"
#include "core/util/include/timer.h"

namespace baidu {
namespace paddle_serving {
namespace serving {

using baidu::paddle_serving::Timer;
using baidu::paddle_serving::predictor::MempoolWrapper;
using baidu::paddle_serving::predictor::general_model::Tensor;
using baidu::paddle_serving::predictor::general_model::Response;
using baidu::paddle_serving::predictor::general_model::Request;
using baidu::paddle_serving::predictor::InferManager;
using baidu::paddle_serving::predictor::PaddleGeneralModelConfig;

int GeneralPicodetOp::inference() {
  VLOG(2) << "Going to run inference";
  const std::vector<std::string> pre_node_names = pre_names();
  if (pre_node_names.size() != 1) {
    LOG(ERROR) << "This op(" << op_name()
               << ") can only have one predecessor op, but received "
               << pre_node_names.size();
    return -1;
  }
  const std::string pre_name = pre_node_names[0];

  const GeneralBlob *input_blob = get_depend_argument<GeneralBlob>(pre_name);
  if (!input_blob) {
    LOG(ERROR) << "input_blob is nullptr,error";
    return -1;
  }
  uint64_t log_id = input_blob->GetLogId();
  VLOG(2) << "(logid=" << log_id << ") Get precedent op name: " << pre_name;

  GeneralBlob *output_blob = mutable_data<GeneralBlob>();
  if (!output_blob) {
    LOG(ERROR) << "output_blob is nullptr,error";
    return -1;
  }
  output_blob->SetLogId(log_id);

  if (!input_blob) {
    LOG(ERROR) << "(logid=" << log_id
               << ") Failed mutable depended argument, op:" << pre_name;
    return -1;
  }

  const TensorVector *in = &input_blob->tensor_vector;
  TensorVector *out = &output_blob->tensor_vector;
  int batch_size = input_blob->_batch_size;
  VLOG(2) << "(logid=" << log_id << ") input batch size: " << batch_size;
  output_blob->_batch_size = batch_size;

  // get image shape
  float *data = (float *)in->at(0).data.data();
  int height = data[0];
  int width = data[1];
  VLOG(2) << "image width: " << width;
  VLOG(2) << "image height: " << height;

  ///////////////////det preprocess begin/////////////////////////
  // read image using opencv
  unsigned char *img_data = static_cast<unsigned char *>(in->at(1).data.data());
  cv::Mat origin(height, width, CV_8UC3, img_data);

  Timer timeline;
  int64_t start = timeline.TimeStampUS();
  timeline.Start();

  ///////////////////skip detection process /////////////////////////
  VLOG(2) << "Skip detection process"
          << "\n";

  ///////////////////det postprocess begin/////////////////////////
  std::vector<ObjectResult> result;

  // 1) add the whole image
  ObjectResult result_whole_img = {{0, 0, width - 1, height - 1}, 0, 1.0};
  result.push_back(result_whole_img);

  // 2) crop image and do preprocess. concanate the data
  cv::Mat srcimg;
  cv::cvtColor(origin, srcimg, cv::COLOR_BGR2RGB);
  std::vector<float> all_data;
  for (int j = 0; j < result.size(); ++j) {
    int w = result[j].rect[2] - result[j].rect[0];
    int h = result[j].rect[3] - result[j].rect[1];
    cv::Rect rect(result[j].rect[0], result[j].rect[1], w, h);
    cv::Mat crop_img = srcimg(rect);
    cv::Mat resize_img;
    resize_op_.Run(crop_img, resize_img, resize_short_, resize_size_);
    normalize_op_.Run(&resize_img, mean_, std_, scale_);
    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    permute_op_.Run(&resize_img, input.data());
    for (int m = 0; m < input.size(); m++) {
      all_data.push_back(input[m]);
    }
  }
  ///////////////////det postprocess begin/////////////////////////

  // generate new Tensors;
  //"x"
  int out_num = all_data.size();
  int databuf_size_out = out_num * sizeof(float);
  void *databuf_data_out = MempoolWrapper::instance().malloc(databuf_size_out);
  if (!databuf_data_out) {
    LOG(ERROR) << "Malloc failed, size: " << databuf_size_out;
    return -1;
  }
  memcpy(databuf_data_out, all_data.data(), databuf_size_out);
  char *databuf_char_out = reinterpret_cast<char *>(databuf_data_out);
  paddle::PaddleBuf paddleBuf_out(databuf_char_out, databuf_size_out);
  paddle::PaddleTensor tensor_out;

  tensor_out.name = "x";
  tensor_out.dtype = paddle::PaddleDType::FLOAT32;
  tensor_out.shape = {result.size(), 3, 224, 224};
  tensor_out.data = paddleBuf_out;
  tensor_out.lod = in->at(0).lod;
  out->push_back(tensor_out);

  //"boxes"
  int box_size_out = result.size() * 6 * sizeof(float);
  void *box_data_out = MempoolWrapper::instance().malloc(box_size_out);
  if (!box_data_out) {
    LOG(ERROR) << "Malloc failed, size: " << box_data_out;
    return -1;
  }
  memcpy(box_data_out, out->at(0).data.data(),
         box_size_out - 6 * sizeof(float));
  float *box_float_out = reinterpret_cast<float *>(box_data_out);
  box_float_out += (result.size() - 1) * 6;
  box_float_out[0] = 0.0;
  box_float_out[1] = 1.0;
  box_float_out[2] = 0.0;
  box_float_out[3] = 0.0;
  box_float_out[4] = width - 1;
  box_float_out[5] = height - 1;
  char *box_char_out = reinterpret_cast<char *>(box_data_out);
  paddle::PaddleBuf paddleBuf_out_2(box_char_out, box_size_out);
  paddle::PaddleTensor tensor_out_2;

  tensor_out_2.name = "boxes";
  tensor_out_2.dtype = paddle::PaddleDType::FLOAT32;
  tensor_out_2.shape = {result.size(), 6};
  tensor_out_2.data = paddleBuf_out_2;
  tensor_out_2.lod = in->at(0).lod;
  out->push_back(tensor_out_2);

  int64_t end = timeline.TimeStampUS();
  CopyBlobInfo(input_blob, output_blob);
  AddBlobInfo(output_blob, start);
  AddBlobInfo(output_blob, end);
  return 0;
}

DEFINE_OP(GeneralPicodetOp);
} // namespace serving
} // namespace paddle_serving
} // namespace baidu
