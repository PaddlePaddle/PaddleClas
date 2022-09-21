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

#include "include/faiss/Index.h"
#include "include/faiss/index_io.h"
#include <cstring>
#include <map>
#include <vector>

struct SearchResult {
  std::vector<faiss::Index::idx_t> I;
  std::vector<float> D;
  int return_k;
};

class VectorSearch {
public:
  explicit VectorSearch(const std::string &label_path,
                        const std::string &index_path, const int &return_k = 5,
                        const float &score_thres = 0.5) {
    //        // IndexProcess
    this->label_path = label_path;
    this->index_path = index_path;
    this->return_k = return_k;
    this->score_thres = score_thres;

    LoadIdMap();
    LoadIndexFile();
    this->I.resize(this->return_k * this->max_query_number);
    this->D.resize(this->return_k * this->max_query_number);
    printf("faiss index load success!\n");
  };

  void LoadIdMap();

  bool LoadFromSaveFileName(const std::string &load_file_name);

  void LoadIndexFile();

  int AddFeature(float *feature, const std::string &label = "");

  const SearchResult &Search(float *feature, int query_number);

  const int GetIndexLength() { return this->index->ntotal; }

  void SaveIndex(const std::string &save_path = "");

  std::string GetIndexPath() { return this->index_path; }

  std::string GetLabelPath() { return this->label_path; }

  std::string GetLabel(faiss::Index::idx_t ind);

  void ClearFeature();

  const float &GetThreshold() const;

  std::vector<std::string> GetLabelList() const;

private:
  std::string index_path;
  std::string label_path;
  int return_k = 5;
  float score_thres = 0.5;

  std::map<long int, std::string> id_map;
  faiss::Index *index;
  int max_query_number = 6;
  std::vector<float> D;
  std::vector<faiss::Index::idx_t> I;
  SearchResult sr;
};
