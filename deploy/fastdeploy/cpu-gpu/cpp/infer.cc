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

#include "fastdeploy/vision.h"
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string &model_dir, const std::string &image_file,
                  const fastdeploy::RuntimeOption &option) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";

  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);
  if (!model.Initialized()) {
    std::cerr << "Failed to initialize." << std::endl;
    return;
  }

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  // print res
  std::cout << res.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: infer_demo path/to/model path/to/image run_option, "
                 "e.g ./infer_demo ./ResNet50_vd ./test.jpeg 0"
              << std::endl;
    return -1;
  }

  fastdeploy::RuntimeOption option;
  int flag = std::atoi(argv[3]);

  if (flag == 0) {
    option.UseCpu();
    option.UsePaddleBackend(); // Paddle Inference
  } else if (flag == 1) {
    option.UseCpu();
    option.UseOpenVINOBackend(); // OpenVINO
  } else if (flag == 2) {
    option.UseCpu();
    option.UseOrtBackend(); // ONNX Runtime
  } else if (flag == 3) {
    option.UseCpu();
    option.UseLiteBackend(); // Paddle Lite
  } else if (flag == 4) {
    option.UseGpu();
    option.UsePaddleBackend(); // Paddle Inference
  } else if (flag == 5) {
    option.UseGpu();
    option.UsePaddleInferBackend();
    option.paddle_infer_option.enable_trt = true;
  } else if (flag == 6) {
    option.UseGpu();
    option.UseOrtBackend(); // ONNX Runtime
  } else if (flag == 7) {
    option.UseGpu();
    option.UseTrtBackend(); // TensorRT
  }

  std::string model_dir = argv[1];
  std::string image_dir = argv[2];
  InitAndInfer(model_dir, image_dir, option);
}
