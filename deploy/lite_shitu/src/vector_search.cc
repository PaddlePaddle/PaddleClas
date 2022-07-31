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

#include "include/vector_search.h"
#include <cstdio>
#include <faiss/index_io.h>
#include <fstream>
#include <iostream>
#include <regex>

namespace PPShiTu {
// load the vector.index
void VectorSearch::LoadIndexFile() {
  std::string file_path = this->index_dir + OS_PATH_SEP + "vector.index";
  const char *fname = file_path.c_str();
  this->index = faiss::read_index(fname, 0);
}

// load id_map.txt
void VectorSearch::LoadIdMap() {
  std::string file_path = this->index_dir + OS_PATH_SEP + "id_map.txt";
  std::ifstream in(file_path);
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      std::regex ws_re("\\s+");
      std::vector<std::string> v(
          std::sregex_token_iterator(line.begin(), line.end(), ws_re, -1),
          std::sregex_token_iterator());
      if (v.size() != 2) {
        std::cout << "The number of element for each line in : " << file_path
                  << "must be 2, exit the program..." << std::endl;
        exit(1);
      } else
        this->id_map.insert(std::pair<long int, std::string>(
            std::stol(v[0], nullptr, 10), v[1]));
    }
  }
}

// doing search
const SearchResult &VectorSearch::Search(float *feature, int query_number) {
  this->D.resize(this->return_k * query_number);
  this->I.resize(this->return_k * query_number);
  this->index->search(query_number, feature, return_k, D.data(), I.data());
  this->sr.return_k = this->return_k;
  this->sr.D = this->D;
  this->sr.I = this->I;
  return this->sr;
}

const std::string &VectorSearch::GetLabel(faiss::Index::idx_t ind) {
  return this->id_map.at(ind);
}

int VectorSearch::AddFeature(float *feature, std::string label) {
  this->index->add(1, feature);
  int id = this->id_map.size();
  if (label != "")
    this->id_map.insert(std::pair<long int, std::string>(id, label));
  else
    this->id_map.insert(
        std::pair<long int, std::string>(id, std::to_string(id)));
  return this->index->ntotal;
}

void VectorSearch::SaveIndex(std::string save_dir) {
  std::string file_path_index, file_path_labelmap;
  if (save_dir == "") {
    file_path_index = this->index_dir + OS_PATH_SEP + "vector.index";
    file_path_labelmap = this->index_dir + OS_PATH_SEP + "id_map.txt";
  } else {
    file_path_index = save_dir + OS_PATH_SEP + "vector.index";
    file_path_labelmap = save_dir + OS_PATH_SEP + "id_map.txt";
  }
  // save index
  faiss::write_index(this->index, file_path_index.c_str());

  // save label_map
  std::ofstream out(file_path_labelmap);
  std::map<long int, std::string>::iterator iter;
  for (iter = this->id_map.begin(); iter != this->id_map.end(); iter++) {
    std::string content = std::to_string(iter->first) + " " + iter->second;
    out.write(content.c_str(), content.size());
    out << std::endl;
  }
  out.close();
}

void VectorSearch::ClearFeature(){
  this->index->reset();
  this->id_map.clear();
}

} // namespace PPShiTu
