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

#include "VectorSearch.h"
#include "Utils.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <regex>
#include <sys/stat.h>
#include <unistd.h>

void VectorSearch::LoadIndexFile() {
  //"/storage/emulated/0/Android/data/com.baidu.paddle.lite.demo.pp_shitu/files/index/vector.index"
  std::string file_path = this->index_path;
  const char *fname = file_path.c_str();
  this->index = faiss::read_index(fname, 0);
}

// load id_map.txt
void VectorSearch::LoadIdMap() {
  std::string file_path = this->label_path;
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

std::string VectorSearch::GetLabel(faiss::Index::idx_t ind) {
  if (this->id_map.count(ind)) {
    return this->id_map[ind];
  } else {
    return "None";
  }
}

int VectorSearch::AddFeature(float *feature, const std::string &label) {
  this->index->add(1, feature);
  int id = (int)(this->id_map.size());
  if (!label.empty()) {
    this->id_map.insert(std::pair<long int, std::string>(id, label));
  } else {
    this->id_map.insert(
        std::pair<long int, std::string>(id, std::to_string(id)));
  }
  return (int)(this->index->ntotal);
}

void VectorSearch::SaveIndex(const std::string &save_file_name) {
  //  save_file_name 为无后缀的文件名字，如 vector、vector_new 等
  std::string file_path_index, file_path_labelmap;
  if (save_file_name.empty()) {
    file_path_index = this->index_path;
    file_path_labelmap = this->label_path;
  } else {
    int begin_pos = (int)this->index_path.find_last_of('/') + 1;
    int end_pos = (int)this->index_path.find_last_of('.');
    int replace_len = end_pos - begin_pos;
    file_path_index =
        this->index_path.replace(begin_pos, replace_len, save_file_name);

    begin_pos = (int)this->label_path.find_last_of('/') + 1;
    end_pos = (int)this->label_path.find_last_of('.');
    replace_len = end_pos - begin_pos;
    file_path_labelmap =
        this->label_path.replace(begin_pos, replace_len, save_file_name);
  }
  // save index
  faiss::write_index(this->index, file_path_index.c_str());
  LOGD("index file saved at [%s]", file_path_index.c_str());

  // save label_map
  std::ofstream out(file_path_labelmap);
  std::map<long int, std::string>::iterator iter;
  for (iter = this->id_map.begin(); iter != this->id_map.end(); iter++) {
    std::string content = std::to_string(iter->first) + " " + iter->second;
    out.write(content.c_str(), (int)content.size());
    out << std::endl;
  }
  out.close();
}

void VectorSearch::ClearFeature() {
  this->index->reset();
  this->id_map.clear();
  LOGD("=========================features cleard");
}

const float &VectorSearch::GetThreshold() const { return this->score_thres; }

bool file_exist(const std::string &file_name) {
  return access(file_name.c_str(), F_OK) != -1;
}

bool VectorSearch::LoadFromSaveFileName(const std::string &load_file_name) {
  std::string origin_label_path = GetLabelPath();
  int begin_pos = (int)origin_label_path.find_last_of('/') + 1;
  int end_pos = (int)origin_label_path.find_last_of('.');
  int replace_len = end_pos - begin_pos;
  std::string new_label_path =
      origin_label_path.replace(begin_pos, replace_len, load_file_name);

  std::string origin_index_path = GetIndexPath();
  begin_pos = (int)origin_index_path.find_last_of('/') + 1;
  end_pos = (int)origin_index_path.find_last_of('.');
  replace_len = end_pos - begin_pos;
  std::string new_index_path =
      origin_index_path.replace(begin_pos, replace_len, load_file_name);

  if (!file_exist(new_label_path) || !file_exist(new_index_path)) {
    return false;
  }
  this->label_path = new_label_path;
  this->id_map.clear();
  LoadIdMap();

  this->index_path = new_index_path;
  LoadIndexFile();
  return true;
}

std::vector<std::string> VectorSearch::GetLabelList() const {
  std::vector<std::string> tmp;
  for (const auto &it : this->id_map) {
    tmp.emplace_back(it.second);
  }
  std::sort(tmp.begin(), tmp.end());
  tmp.erase(unique(tmp.begin(), tmp.end()), tmp.end());
  return tmp;
}
