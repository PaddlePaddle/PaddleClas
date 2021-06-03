# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from interface import Graph_Index

# 随机产生样本
index_vectors = np.random.rand(100000,128).astype(np.float32) 
query_vector = np.random.rand(128).astype(np.float32) 
index_docs = ["ID_"+str(i) for i in range(100000)]

# 初始化索引结构
indexer = Graph_Index(dist_type="IP") #支持"IP"和"L2"
indexer.build(gallery_vectors=index_vectors, gallery_docs=index_docs, pq_size=100, index_path='test')

# 查询
scores, docs = indexer.search(query=query_vector, return_k=10, search_budget=100)
print(scores)
print(docs)

# 保存与加载
indexer.dump(index_path="test") 
indexer.load(index_path="test") 
