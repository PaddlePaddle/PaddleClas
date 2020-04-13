# 多机训练

分布式训练的高性能，是飞桨的核心优势技术之一，在分类任务上，分布式训练可以达到几乎线性的加速比。
[Fleet](https://github.com/PaddlePaddle/Fleet) 是用于 PaddlePaddle 分布式训练的高层 API，基于这套接口用户可以很容易切换到分布式训练程序。
为了可以同时支持单机训练和多机训练，[PaddleClas](https://github.com/PaddlePaddle/PaddleClas) 采用 Fleet API 接口，更多的分布式训练可以参考 [Fleet API设计文档](https://github.com/PaddlePaddle/Fleet/blob/develop/README.md)。

