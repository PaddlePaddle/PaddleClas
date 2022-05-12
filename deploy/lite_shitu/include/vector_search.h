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

#include "json/json.h"
#include <cstring>
#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <map>

namespace PPShiTu {
struct SearchResult {
  std::vector<faiss::Index::idx_t> I;
  std::vector<float> D;
  int return_k;
};

class VectorSearch {
public:
  explicit VectorSearch(const Json::Value &config) {
    // IndexProcess
    this->index_dir = config["IndexProcess"]["index_dir"].as<std::string>();
    this->return_k = config["IndexProcess"]["return_k"].as<int>();
    this->score_thres = config["IndexProcess"]["score_thres"].as<float>();
    this->max_query_number = config["Global"]["max_det_results"].as<int>() + 1;

    LoadIdMap();
    LoadIndexFile();
    this->I.resize(this->return_k * this->max_query_number);
    this->D.resize(this->return_k * this->max_query_number);
    printf("faiss index load success!\n");
  };

  void LoadIdMap();

  void LoadIndexFile();

  const SearchResult &Search(float *feature, int query_number);

  const std::string &GetLabel(faiss::Index::idx_t ind);

  const float &GetThreshold() { return this->score_thres; }

private:
  std::string index_dir;
  int return_k = 5;
  float score_thres = 0.5;

  std::map<long int, std::string> id_map;
  faiss::Index *index;
  int max_query_number = 6;
  std::vector<float> D;
  std::vector<faiss::Index::idx_t> I;
  SearchResult sr;
};
}
