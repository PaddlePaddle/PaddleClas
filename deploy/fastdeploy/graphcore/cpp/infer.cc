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

void IPUInfer(const std::string &model_dir, const std::string &image_file) {
  auto model_file = model_dir + sep + "inference.pdmodel";
  auto params_file = model_dir + sep + "inference.pdiparams";
  auto config_file = model_dir + sep + "inference_cls.yaml";

  auto option = fastdeploy::RuntimeOption();
  option.UseIPU();

  auto model = fastdeploy::vision::classification::PaddleClasModel(
      model_file, params_file, config_file, option);

  assert(model.Initialized());

  auto im = cv::imread(image_file);

  fastdeploy::vision::ClassifyResult res;
  if (!model.Predict(&im, &res)) {
    std::cerr << "Failed to predict." << std::endl;
    return;
  }

  std::cout << res.Str() << std::endl;
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: infer_demo path/to/model path/to/image " << std::endl;
    return -1;
  }

  std::string model_dir = argv[1];
  std::string test_image = argv[2];
  IPUInfer(model_dir, test_image);
  return 0;
}
