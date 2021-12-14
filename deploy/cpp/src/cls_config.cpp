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
#include <ostream>

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

    void ClsConfig::ReadLabelMap() {
        if (this->class_id_map_path.empty()) {
            std::cout << "The Class Label file dose not input" << std::endl;
            return;
        }
        std::ifstream in(this->class_id_map_path);
        std::string line;
        if (in) {
            while (getline(in, line)) {
                int split_flag = line.find_first_of(" ");
                this->id_map[std::stoi(line.substr(0, split_flag))] =
                        line.substr(split_flag + 1, line.size());
            }
        }
    }
}; // namespace PaddleClas
