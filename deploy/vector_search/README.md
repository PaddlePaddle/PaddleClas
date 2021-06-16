# 向量检索


## 简介

一些垂域识别任务（如车辆、商品等）需要识别的类别数较大，往往采用基于检索的方式，通过查询向量与底库向量进行快速的最近邻搜索，获得匹配的预测类别。向量检索模块提供基础的近似最近邻搜索算法，基于百度自研的Möbius算法，一种基于图的近似最近邻搜索算法，用于最大内积搜索 (MIPS)。 该模块提供python接口，支持numpy和 tensor类型向量,支持L2和Inner Product距离计算。

Mobius 算法细节详见论文 （[Möbius Transformation for Fast Inner Product Search on Graph](http://research.baidu.com/Public/uploads/5e189d36b5cf6.PDF), [Code](https://github.com/sunbelbd/mobius)）



## 安装

若index.so不可用，在项目目录下运行以下命令生成新的index.so文件

    make index.so

编译环境:  g++ 5.4.0 , 9.3.0.  其他版本也可能工作。 请确保您的 C++ 编译器支持 C++11 标准。



## 快速使用

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
