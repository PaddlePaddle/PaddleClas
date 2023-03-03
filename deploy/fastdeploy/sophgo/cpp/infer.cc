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
#include <string>
#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

void InitAndInfer(const std::string &model_dir, const std::string &image_file) {
  auto model_file = model_dir + sep + "resnet50_1684x_f32.bmodel";
  auto params_file = model_dir + sep + "";
  auto config_file = model_dir + sep + "preprocess_config.yaml";

  fastdeploy::RuntimeOption option;
  option.UseSophgo();
  auto model_format = fastdeploy::ModelFormat::SOPHGO;
  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option, model_format);

  assert(model.Initialized());

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: infer_demo path/to/model "
                 "path/to/image "
                 "run_option, "
                 "e.g ./infer_demo ./bmodel ./test.jpeg"
              << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  InitAndInfer(model_dir, test_image);
  return 0;
}
