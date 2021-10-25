# 版本更新信息

## v2.3

- 模型更新
  - 添加轻量化模型预训练权重，包括检测模型、特征模型
  - 发布PP-LCNet系列模型，此系列模型是专门在CPU上设计运行的自研模型
  - SwinTransformer、Twins、Deit支持从scrach直接训练，达到论文精度
- 框架基础能力
  - 添加DeepHash模块，支持特征模型直接输出二值特征
  - 添加PKSampler，特征模型不能多机多卡的训练的问题
  - 支持PaddleSlim：支持分类模型、特征模型的量化、裁剪训练及离线量化功能
  - Legendary models支持模型中间结果输出
  - 支持多标签分类训练
- 预测部署
  - 使用Faiss替换原有特征检索库，提升平台适配性
  - 支持PaddleServing：支持分类模型、图像识别流程的部署

- 推荐库版本
  - python版本：3.7
  - PaddlePaddle版本：2.1.3
  - PaddleSlim版本：2.2.0
  - PaddleServing版本：0.6.1

## v2.2

- 模型更新
  - 添加LeViT、Twins、TNT、DLA、HardNet、RedNet、SwinTransfomer模型
- 框架基础能力
  - 将分类模型分为两类
    -  legendary models：引入TheseusLayer基类，及增加了修改网络功能接口，同时支持网络截断输出功能
    - model zoo：其他普通分类模型
  - 添加Metric Learning算法支持
    - 添加多种相关Loss算法，及基础网络模块gears（支持与backbone、loss组合）方便使用
    - 同时支持普通分类及metric learning相关任务训练
- 应用更新
  - 添加商品识别、车辆识别（车辆细粒度分类、车辆ReID）、logo识别、动漫人物识别应用具体案例及相关模型
  - 添加图像识别完整pipeline，包含检测模块、特征提取模块、向量检索模块
- 预测部署
  - 添加百度自研向量检索模块Mobius，支持图像识别系统预测部署
- 文档更新
  - 添加图像识别相关文档
  - 修复之前文档bug
- 推荐库版本
  - python版本：3.7
  - PaddlePaddle：2.1.2
