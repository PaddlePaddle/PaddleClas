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

#include <iostream>
#include <ostream>
#include <vector>

#include <include/yaml_config.h>


std::vector <std::string> YamlConfig::ReadDict(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::vector <std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    } else {
        std::cout << "no such label file: " << path << ", exit the program..."
                  << std::endl;
        exit(1);
    }
    return m_vec;
}

std::map<int, std::string> YamlConfig::ReadIndexId(const std::string &path) {
    std::ifstream in(path);
    std::string line;
    std::map<int, std::string> m_vec;
    if (in) {
        while (getline(in, line)) {
            std::regex ws_re("\\s+");
            std::vector <std::string> v(
                    std::sregex_token_iterator(line.begin(), line.end(), ws_re, -1),
                    std::sregex_token_iterator());
            if (v.size() != 3) {
                std::cout << "The number of element for each line in : " << path
                          << "must be 3, exit the program..." << std::endl;
                exit(1);
            } else
                m_vec.insert(std::pair<int, std::string>(stoi(v[0]), v[2]));
        }
    }
    return m_vec;
}

YAML::Node YamlConfig::ReadYamlConfig(const std::string &path) {
    YAML::Node config;
    try {
        config = YAML::LoadFile(path);
    } catch (YAML::BadFile &e) {
        std::cout << "Something wrong in yaml file, please check yaml file"
                  << std::endl;
        exit(1);
    }
    return config;
}

void YamlConfig::PrintConfigInfo() {
    std::cout << this->config_file << std::endl;
    //   for (YAML::const_iterator
    //   it=config_file.begin();it!=config_file.end();++it)
    // {
    //   std::cout << it->as<std::string>() << "\n";
    //   }
}
