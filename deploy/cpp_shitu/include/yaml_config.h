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

#pragma once

#ifdef WIN32
#define OS_PATH_SEP "\\"
#else
#define OS_PATH_SEP "/"
#endif

#include <chrono>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <stdlib.h>
#include <vector>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <regex>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "yaml-cpp/yaml.h"


class YamlConfig {
public:
    explicit YamlConfig(const std::string &path) {
        config_file = ReadYamlConfig(path);
    }

    static std::vector <std::string> ReadDict(const std::string &path);

    static std::map<int, std::string> ReadIndexId(const std::string &path);

    static YAML::Node ReadYamlConfig(const std::string &path);

    void PrintConfigInfo();

    YAML::Node config_file;
};
