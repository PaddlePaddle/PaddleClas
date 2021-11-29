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

#include <include/cls_config.h>

namespace PaddleClas {

void ClsConfig::PrintConfigInfo() {
  std::cout << "=======Paddle Class inference config======" << std::endl;
  std::cout << this->config_file << std::endl;
  std::cout << "=======End of Paddle Class inference config======" << std::endl;
}

void ClsConfig::ReadYamlConfig(const std::string &path) {

  try {
    this->config_file = YAML::LoadFile(path);
  } catch (YAML::BadFile &e) {
    std::cout << "Something wrong in yaml file, please check yaml file"
              << std::endl;
    exit(1);
  }
}
}; // namespace PaddleClas
